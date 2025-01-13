import sys
from pathlib import Path
import os
import threading
import time
import re
import traceback
import subprocess
import shutil
from queue import Queue
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .generate_utils import get_config, set_config, get_data_path, get_embedded_python_path
from .generate_command import generate_command
from .post_process import post_process_midi

# --------------------------
# Logger Setup & Redirection
# --------------------------
class LoggerWriter:
    """
    A helper class that allows us to redirect Python's stdout and stderr 
    to the standard Python logging system.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():  # Ignore empty messages
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass
    
    def isatty(self):
        return False  # Indicate this is not a terminal

def setup_logging(log_file="app.log"):
    """
    Set up logging to capture all stdout and stderr messages.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # Also output to console (overridden by LoggerWriter)
        ]
    )

    # Redirect stdout and stderr to our LoggerWriter
    sys.stdout = LoggerWriter(logging.getLogger(), logging.INFO)
    sys.stderr = LoggerWriter(logging.getLogger(), logging.ERROR)

# --------------------
# The MIDI_Generator
# --------------------
class MIDI_Generator:
    def __init__(self):
        python_path = get_embedded_python_path()

        def install_requirements(requirements_file):
            """
            Install requirements from a given requirements.txt file and 
            handle system dependencies if needed.
            """
            requirements_file = get_data_path("requirements.txt")

            def check_and_install_brew_dependency(dependency):
                """
                Check if a brew dependency is installed, and install it if missing.
                """
                if shutil.which("brew") is None:
                    print("Homebrew is not installed. Please install Homebrew first.")
                    sys.exit(1)
                
                try:
                    result = subprocess.run(["brew", "list", dependency], 
                                            stdout=subprocess.PIPE, 
                                            stderr=subprocess.PIPE, text=True)
                    if result.returncode != 0:  # Dependency not installed
                        print(f"Installing {dependency} with Homebrew...")
                        subprocess.check_call(["brew", "install", dependency])
                        print(f"{dependency} installed successfully.")
                    else:
                        print(f"{dependency} is already installed.")
                except Exception as e:
                    print(f"Error while checking/installing {dependency}: {e}")
                    sys.exit(1)

            # Define system dependencies required by specific Python packages
            brew_dependencies = {
                "PyQt5": ["qt"],  # Example if needed
                "pretty_midi": ["ffmpeg", "libsndfile"],  # For audio-related operations
                # Add more as needed
            }

            print(f"Installing requirements from {os.path.abspath(requirements_file)}...")

            # Check and install required brew dependencies if they appear in requirements
            for package, dependencies in brew_dependencies.items():
                with open(requirements_file, "r") as req_file:
                    if any(package in line for line in req_file):
                        for dependency in dependencies:
                            check_and_install_brew_dependency(dependency)

            # Install Python requirements
            try:
                subprocess.check_call([python_path, "-m", "pip", "install", "-r", requirements_file])
                print(f"All requirements from {requirements_file} installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error during pip installation: {e}")
                sys.exit(1)

        # Attempt to install missing Python packages
        install_requirements("requirements.txt")

        # Default parameters you want to expose in the UI
        self.parameters = {
            "prompt": {"type": "text", "default": "A happy and groovy melody.", "label": "Prompt", "required": True},
            "key": {"type": "text", "default": "C Major", "label": "Key", "required": True},
            "num_bars": {"type": "number", "default": 8, "label": "Number of Bars", "required": False},
            "gen_len": {"type": "number", "default": 400, "label": "Generation Length (tokens)"},
            "max_input_len": {"type": "number", "default": 400, "label": "Max Input Length (tokens)"},
            "temp_notes": {"type": "number", "default": 2.1, "label": "Temperature (Notes)"},
            "temp_rests": {"type": "number", "default": 2.1, "label": "Temperature (Rests)"},
            "topk": {"type": "number", "default": -1, "label": "Top-K Sampling"},
            "topp": {"type": "number", "default": 0.7, "label": "Top-P (Nucleus) Sampling"},
            "penalty_coeff": {"type": "number", "default": 0.5, "label": "Penalty Coefficient"},
            "no_amp": {"type": "checkbox", "default": False, "label": "Disable Automatic Mixed Precision"},
            "prune": {"type": "checkbox", "default": True, "label": "Prune Tokens (20%)"},
            "no_cuda": {"type": "checkbox", "default": True, "label": "Use CPU (No CUDA/MPS)"},
            "counter_melody_hyperness": {"type": "slider", "default": 0.5, "label": "Counter Melody Hyperness", "required": False}
        }

        # Retrieve config for output/log paths
        config = get_config()
        self.output = config.get("output", "")
        self.log_path = os.path.abspath(os.path.join(self.output, "..", "logs", "generation.log")) if self.output else "app.log"
        
        # Set up logging
        setup_logging(self.log_path)

        if self.output:
            set_config("log_path", self.log_path)
            print(f"Output directory: {self.output}")
            print(f"Log path: {self.log_path}")
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        else:
            # No output set yet
            self.log_path = ""

    def validate_inputs(self, args):
        """Validate all input parameters based on AI assistance."""
        required_fields = [k for k, v in self.parameters.items() if v.get("required", False)]
        
        if args.get("use_ai"):
            # If AI assistance is enabled, 'prompt' is required
            fields_to_check = required_fields
        else:
            # If AI assistance is disabled, 'valence' and 'arousal' are also required
            fields_to_check = required_fields + ['valence', 'arousal']
        
        missing = []
        for field in fields_to_check:
            if field not in args or args[field] in [None, '']:
                missing.append(field)
        
        if missing:
            return False, missing
        return True, []

    def monitor_log_progress(self, progress_queue):
        """
        A background method that scans the log file for lines indicating 
        token generation progress and sends that progress back via the queue.
        """
        token_pattern = re.compile(r"Generating token (\d+)/(\d+)")
        last_position = 0

        while True:
            try:
                # Ensure log directory/file exist
                if self.log_path:
                    os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                    if not os.path.exists(self.log_path):
                        with open(self.log_path, "w") as f:
                            f.write("")
                    
                    with open(self.log_path, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()

                        if new_lines:  
                            for line in new_lines:
                                token_match = token_pattern.search(line)
                                if token_match:
                                    current_token = int(token_match.group(1))
                                    total_tokens = int(token_match.group(2))
                                    progress = current_token / total_tokens
                                    progress_queue.put(
                                        ("progress", progress, f"Generating token {current_token}/{total_tokens}")
                                    )
                                progress_queue.put(("log", line.strip()))
                time.sleep(0.1)
            except Exception as e:
                progress_queue.put(("error", f"Error reading log file: {str(e)}"))
                break

    def get_subprocess_kwargs(self):
        """
        Returns a dictionary of keyword arguments for subprocess.Popen
        to suppress new windows on Windows and handle macOS appropriately.
        """
        kwargs = {
            'stdout': subprocess.PIPE,
            'stderr': subprocess.PIPE,
            'text': True,
        }
        if os.name == 'nt':
            # Windows-specific: Suppress the console window
            CREATE_NO_WINDOW = 0x08000000
            kwargs['creationflags'] = CREATE_NO_WINDOW
        elif sys.platform == 'darwin':
            # macOS-specific: no additional flags needed
            pass
        else:
            # For other UNIX-like systems, no additional flags are typically required
            pass
        return kwargs

    def generate_midi(self, args, progress_queue):
        """
        Generate a MIDI using an external script, then post-process it.
        """
        try:
            # Start log monitoring in a separate thread
            log_monitor = threading.Thread(target=self.monitor_log_progress, args=(progress_queue,))
            log_monitor.daemon = True
            log_monitor.start()
            
            # Define processed directory
            processed_dir = os.path.join(args.get("out_dir"), "processed_midis")
            os.makedirs(processed_dir, exist_ok=True)

            model_dir = get_data_path("models/continuous_concat")
            if not os.path.isdir(model_dir):
                progress_queue.put(("error", f"Model directory not found: {model_dir}"))
                return

            # Call the generate_command function
            try:
                generate_command(args)
            except Exception as e:
                error_traceback = traceback.format_exc()
                progress_queue.put(("error", f"Error during MIDI generation: {e}\n{error_traceback}"))
                print(f"Error during MIDI generation: {e}\n{error_traceback}")
                return
            
            # After generation, perform post-processing
            generated_midis = [f for f in os.listdir(args.get("out_dir")) if f.endswith(".mid")]
            if not generated_midis:
                progress_queue.put(("error", "No MIDI files generated."))
                return

            latest_midi = max(
                [os.path.join(args.get("out_dir"), f) for f in generated_midis], 
                key=os.path.getmtime
            )
            progress_queue.put(("info", f"Processing: {latest_midi}"))

            # Call post_process_midi
            post_process_midi(
                input_file=latest_midi,
                output_dir=processed_dir,
                key=args.get("key"),
                num_bars=args.get("num_bars"),
                counter_hyperness=args.get("counter_melody_hyperness", 0.5)
            )

            progress_queue.put(("success", "Generation and post-processing completed successfully."))
        
        except Exception as e:
            progress_queue.put(("error", f"Error during generation: {str(e)}"))
            traceback.print_exc()

    def post_process(self, input_dir, output_dir, progress_queue, key, num_bars, counter_melody_hyperness):
        """Run post-processing on generated MIDI files."""
        try:
            midi_files = [f for f in os.listdir(input_dir) if f.endswith(".mid")]
            if not midi_files:
                progress_queue.put(("error", f"No MIDI files found in {input_dir}"))
                return

            latest_midi = max([os.path.join(input_dir, f) for f in midi_files], key=os.path.getmtime)
            progress_queue.put(("info", f"Processing: {latest_midi}"))

            post_process_script = get_data_path("src/midi_app/post_process.py")
            if not os.path.exists(post_process_script):
                progress_queue.put(("error", f"post_process.py not found at {post_process_script}"))
                return

            python_path = get_embedded_python_path()
            command = [
                python_path, post_process_script,
                "--input", latest_midi,
                "--output_dir", output_dir,
                "--num_bars", str(num_bars),
                "--counter_hyperness", str(counter_melody_hyperness),
                "--key", key
            ]

            kwargs = self.get_subprocess_kwargs()

            process = subprocess.Popen(command, **kwargs)
            stdout_lines = []
            stderr_lines = []

            while True:
                stdout_output = process.stdout.readline()
                stderr_output = process.stderr.readline()

                if stdout_output:
                    progress_queue.put(("info", stdout_output.strip()))
                    stdout_lines.append(stdout_output.strip())

                if stderr_output:
                    progress_queue.put(("error", stderr_output.strip()))
                    stderr_lines.append(stderr_output.strip())

                if process.poll() is not None:
                    break

            process.wait()
            if process.returncode != 0:
                stderr_message = "\n".join(stderr_lines)
                progress_queue.put((
                    "error",
                    f"Post-processing failed with return code {process.returncode}. "
                    f"Error details:\n{stderr_message}"
                ))
                return

            progress_queue.put(("success", "Processing completed successfully"))

        except Exception as e:
            progress_queue.put(("error", f"Error during post-processing: {str(e)}"))

# ------------------------
# Main Tkinter Application
# ------------------------
class MainApplication(tk.Tk):
    """Main Application Window using Tkinter."""

    def __init__(self):
        super().__init__()
        self.title("AI MIDI Generation Interface")
        self.geometry("950x700")
        
        # Create a MIDI_Generator instance
        self.generator = MIDI_Generator()
        
        # The queue and thread for background processing
        self.progress_queue = None
        self.thread = None

        # Prepare the UI
        self._init_ui()

        # We will poll the queue at regular intervals
        self.poll_interval_ms = 100  # 100 ms

    def _init_ui(self):
        """Initialize all UI components."""
        main_frame = ttk.Frame(self)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 1) Output Directory Section
        output_frame = ttk.LabelFrame(main_frame, text="Output Directory")
        output_frame.pack(fill='x', pady=5)

        self.output_label_var = tk.StringVar(
            value="Not set" if not self.generator.output else self.generator.output
        )
        output_label = ttk.Label(output_frame, textvariable=self.output_label_var)
        output_label.pack(side='left', padx=5, pady=5)

        set_output_btn = ttk.Button(output_frame, text="Set Output Directory", command=self.set_output_directory)
        set_output_btn.pack(side='right', padx=5, pady=5)

        # 2) Configuration Section
        config_frame = ttk.LabelFrame(main_frame, text="Configuration")
        config_frame.pack(fill='both', expand=True, pady=5)

        # Make a canvas+scrollbar inside config_frame for scrolling
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # This binding ensures the canvas scroll region updates when the inner frame changes size
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # We'll keep a row index to place widgets in a grid
        row_idx = 0

        # -- AI Assistance Checkbox --
        self.use_ai_var = tk.BooleanVar(value=False)
        use_ai_checkbox = ttk.Checkbutton(
            scrollable_frame,
            text="Let AI Control Parameters (Valence, Arousal, Temps) Based on Prompt",
            variable=self.use_ai_var,
            command=self.toggle_ai_assistance
        )
        use_ai_checkbox.grid(row=row_idx, column=0, columnspan=2, sticky='w', pady=5)
        row_idx += 1

        # Prompt
        self.prompt_label = ttk.Label(scrollable_frame, text="Prompt:")
        self.prompt_label.grid(row=row_idx, column=0, padx=5, sticky='e')
        self.prompt_var = tk.StringVar(value=self.generator.parameters["prompt"]["default"])
        self.prompt_entry = ttk.Entry(scrollable_frame, textvariable=self.prompt_var, width=50)
        self.prompt_entry.grid(row=row_idx, column=1, padx=5, pady=2, sticky='w')
        row_idx += 1

        # Key
        key_label = ttk.Label(scrollable_frame, text="Key:")
        key_label.grid(row=row_idx, column=0, padx=5, sticky='e')
        self.key_var = tk.StringVar(value=self.generator.parameters["key"]["default"])
        key_entry = ttk.Entry(scrollable_frame, textvariable=self.key_var, width=50)
        key_entry.grid(row=row_idx, column=1, padx=5, pady=2, sticky='w')
        row_idx += 1

        # Number of Bars
        num_bars_label = ttk.Label(scrollable_frame, text="Number of Bars:")
        num_bars_label.grid(row=row_idx, column=0, padx=5, sticky='e')
        self.num_bars_var = tk.IntVar(value=self.generator.parameters["num_bars"]["default"])
        num_bars_spin = ttk.Spinbox(scrollable_frame, from_=1, to=9999, textvariable=self.num_bars_var)
        num_bars_spin.grid(row=row_idx, column=1, padx=5, pady=2, sticky='w')
        row_idx += 1

        # Generation Length
        gen_len_label = ttk.Label(scrollable_frame, text="Generation Length (tokens):")
        gen_len_label.grid(row=row_idx, column=0, padx=5, sticky='e')
        self.gen_len_var = tk.IntVar(value=self.generator.parameters["gen_len"]["default"])
        gen_len_spin = ttk.Spinbox(scrollable_frame, from_=1, to=999999, textvariable=self.gen_len_var)
        gen_len_spin.grid(row=row_idx, column=1, padx=5, pady=2, sticky='w')
        row_idx += 1

        # Max Input Length
        max_input_len_label = ttk.Label(scrollable_frame, text="Max Input Length (tokens):")
        max_input_len_label.grid(row=row_idx, column=0, padx=5, sticky='e')
        self.max_input_len_var = tk.IntVar(value=self.generator.parameters["max_input_len"]["default"])
        max_input_len_spin = ttk.Spinbox(scrollable_frame, from_=1, to=999999, textvariable=self.max_input_len_var)
        max_input_len_spin.grid(row=row_idx, column=1, padx=5, pady=2, sticky='w')
        row_idx += 1

        # Temperature (Notes)
        temp_notes_label = ttk.Label(scrollable_frame, text="Temperature (Notes):")
        temp_notes_label.grid(row=row_idx, column=0, padx=5, sticky='e')
        self.temp_notes_var = tk.DoubleVar(value=self.generator.parameters["temp_notes"]["default"])
        temp_notes_spin = ttk.Spinbox(
            scrollable_frame, from_=0.0, to=10.0, increment=0.1, textvariable=self.temp_notes_var
        )
        temp_notes_spin.grid(row=row_idx, column=1, padx=5, pady=2, sticky='w')
        row_idx += 1

        # Temperature (Rests)
        temp_rests_label = ttk.Label(scrollable_frame, text="Temperature (Rests):")
        temp_rests_label.grid(row=row_idx, column=0, padx=5, sticky='e')
        self.temp_rests_var = tk.DoubleVar(value=self.generator.parameters["temp_rests"]["default"])
        temp_rests_spin = ttk.Spinbox(
            scrollable_frame, from_=0.0, to=10.0, increment=0.1, textvariable=self.temp_rests_var
        )
        temp_rests_spin.grid(row=row_idx, column=1, padx=5, pady=2, sticky='w')
        row_idx += 1

        # Top-K
        topk_label = ttk.Label(scrollable_frame, text="Top-K Sampling:")
        topk_label.grid(row=row_idx, column=0, padx=5, sticky='e')
        self.topk_var = tk.IntVar(value=self.generator.parameters["topk"]["default"])
        topk_spin = ttk.Spinbox(scrollable_frame, from_=-1, to=999999, textvariable=self.topk_var)
        topk_spin.grid(row=row_idx, column=1, padx=5, pady=2, sticky='w')
        row_idx += 1

        # Top-P
        topp_label = ttk.Label(scrollable_frame, text="Top-P (Nucleus) Sampling:")
        topp_label.grid(row=row_idx, column=0, padx=5, sticky='e')
        self.topp_var = tk.DoubleVar(value=self.generator.parameters["topp"]["default"])
        topp_spin = ttk.Spinbox(scrollable_frame, from_=0.0, to=1.0, increment=0.01, textvariable=self.topp_var)
        topp_spin.grid(row=row_idx, column=1, padx=5, pady=2, sticky='w')
        row_idx += 1

        # Penalty Coefficient
        penalty_label = ttk.Label(scrollable_frame, text="Penalty Coefficient:")
        penalty_label.grid(row=row_idx, column=0, padx=5, sticky='e')
        self.penalty_var = tk.DoubleVar(value=self.generator.parameters["penalty_coeff"]["default"])
        penalty_spin = ttk.Spinbox(scrollable_frame, from_=0.0, to=10.0, increment=0.1, textvariable=self.penalty_var)
        penalty_spin.grid(row=row_idx, column=1, padx=5, pady=2, sticky='w')
        row_idx += 1

        # Checkboxes
        self.no_amp_var = tk.BooleanVar(value=self.generator.parameters["no_amp"]["default"])
        no_amp_cb = ttk.Checkbutton(scrollable_frame, text="Disable Automatic Mixed Precision", variable=self.no_amp_var)
        no_amp_cb.grid(row=row_idx, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        row_idx += 1

        self.prune_var = tk.BooleanVar(value=self.generator.parameters["prune"]["default"])
        prune_cb = ttk.Checkbutton(scrollable_frame, text="Prune Tokens (20%)", variable=self.prune_var)
        prune_cb.grid(row=row_idx, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        row_idx += 1

        self.no_cuda_var = tk.BooleanVar(value=self.generator.parameters["no_cuda"]["default"])
        no_cuda_cb = ttk.Checkbutton(scrollable_frame, text="Use CPU (No CUDA/MPS)", variable=self.no_cuda_var)
        no_cuda_cb.grid(row=row_idx, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        row_idx += 1

        # --- Counter Melody Hyperness ---
        # Define the label and value label BEFORE creating the Scale
        counter_label = ttk.Label(scrollable_frame, text="Counter Melody Hyperness (0.0=calm,1.0=hyper):")
        counter_label.grid(row=row_idx, column=0, padx=5, sticky='e')

        # Initialize the value label first
        self.counter_value_label = ttk.Label(scrollable_frame, text=f"{self.generator.parameters['counter_melody_hyperness']['default']:.2f}")
        self.counter_value_label.grid(row=row_idx, column=1, padx=5, pady=2, sticky='w')
        row_idx += 1

        # Now create the Scale widget
        self.counter_var_label = tk.DoubleVar(value=self.generator.parameters["counter_melody_hyperness"]["default"])
        self.counter_slider = ttk.Scale(
            scrollable_frame, from_=0.0, to=1.0, 
            orient=tk.HORIZONTAL,
            command=self.on_counter_changed
        )
        self.counter_slider.set(self.counter_var_label.get())
        self.counter_slider.grid(row=row_idx, column=1, padx=5, pady=2, sticky='we')
        row_idx += 1

        # --- Additional Valence/Arousal if AI is disabled ---
        # Define valence and arousal labels and value labels BEFORE creating the Scales
        self.valence_label = ttk.Label(scrollable_frame, text="Valence:")
        self.arousal_label = ttk.Label(scrollable_frame, text="Arousal:")

        # Initialize the valence value label first
        self.valence_value_label = ttk.Label(scrollable_frame, text=f"{0.5:.2f}")
        self.valence_value_label.grid(row=row_idx, column=1, padx=5, sticky='w')
        row_idx += 1

        # Create the valence Scale widget
        self.valence_var_label = tk.DoubleVar(value=0.5)
        self.valence_slider = ttk.Scale(
            scrollable_frame, from_=0.0, to=3.0, 
            orient=tk.HORIZONTAL, 
            command=self.on_valence_changed
        )
        self.valence_slider.set(self.valence_var_label.get())
        self.valence_slider.grid(row=row_idx, column=1, padx=5, sticky='we')
        self.valence_label.grid(row=row_idx, column=0, padx=5, sticky='e')
        row_idx += 1

        # Initialize the arousal value label first
        self.arousal_value_label = ttk.Label(scrollable_frame, text=f"{0.5:.2f}")
        self.arousal_value_label.grid(row=row_idx, column=1, padx=5, sticky='w')
        row_idx += 1

        # Create the arousal Scale widget
        self.arousal_var = tk.DoubleVar(value=0.5)
        self.arousal_slider = ttk.Scale(
            scrollable_frame, from_=0.0, to=3.0, 
            orient=tk.HORIZONTAL, 
            command=self.on_arousal_changed
        )
        self.arousal_slider.set(self.arousal_var.get())
        self.arousal_slider.grid(row=row_idx, column=1, padx=5, sticky='we')
        self.arousal_label.grid(row=row_idx, column=0, padx=5, sticky='e')
        row_idx += 1

        # 3) Progress & Logs
        progress_frame = ttk.LabelFrame(main_frame, text="Progress & Logs")
        progress_frame.pack(fill='both', expand=True, pady=5)

        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress_bar.pack(fill='x', padx=5, pady=5)

        self.status_label_var = tk.StringVar(value="Idle")
        status_label = ttk.Label(progress_frame, textvariable=self.status_label_var)
        status_label.pack(pady=5)

        self.logs_text = tk.Text(progress_frame, wrap='word', height=10)
        self.logs_text.pack(fill='both', expand=True, padx=5, pady=5)

        self.errors_label_var = tk.StringVar(value="")
        self.errors_label = ttk.Label(progress_frame, textvariable=self.errors_label_var, foreground='red')
        self.errors_label.pack(pady=5)

        # 4) Generate Button
        generate_btn = ttk.Button(main_frame, text="Generate MIDI", command=self.on_generate_clicked)
        generate_btn.pack(pady=5)

        # Toggle initial visibility (hide valence/arousal if AI is on)
        self.toggle_ai_assistance()

    def set_output_directory(self):
        """Set the output directory via a folder dialog."""
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            try:
                output_path = Path(dir_path).resolve()
                output_path.mkdir(parents=True, exist_ok=True)

                set_config("output", str(output_path))
                
                # Update log path too
                default_log_path = str(output_path / "logs" / "generation.log")
                set_config("log_path", default_log_path)

                self.generator.output = str(output_path)
                self.generator.log_path = default_log_path

                self.output_label_var.set(str(output_path))
                
                messagebox.showinfo("Success", f"Output directory set to: {output_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to set output directory: {e}")

    def toggle_ai_assistance(self):
        """Hide or show valence/arousal based on AI assistance checkbox."""
        use_ai = self.use_ai_var.get()
        # Enable or disable the prompt if AI is used
        if use_ai:
            self.prompt_entry.configure(state='normal')
        else:
            self.prompt_entry.configure(state='disabled')

        # Show valence/arousal only if NOT using AI
        if not use_ai:
            self.valence_label.grid()
            self.valence_slider.grid()
            self.valence_value_label.grid()
            self.arousal_label.grid()
            self.arousal_slider.grid()
            self.arousal_value_label.grid()
        else:
            self.valence_label.grid_remove()
            self.valence_slider.grid_remove()
            self.valence_value_label.grid_remove()
            self.arousal_label.grid_remove()
            self.arousal_slider.grid_remove()
            self.arousal_value_label.grid_remove()

    def on_counter_changed(self, val):
        """Update display label when counter hyperness slider changes."""
        try:
            val_float = float(val)
            self.counter_var_label.set(val_float)
            self.counter_value_label.config(text=f"{val_float:.2f}")
        except AttributeError as e:
            print(f"[ERROR] AttributeError in on_counter_changed: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error in on_counter_changed: {e}")

    def on_valence_changed(self, val):
        """Update display label when valence slider changes."""
        try:
            val_float = float(val)
            self.valence_var_label.set(val_float)
            self.valence_value_label.config(text=f"{val_float:.2f}")
        except AttributeError as e:
            print(f"[ERROR] AttributeError in on_valence_changed: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error in on_valence_changed: {e}")

    def on_arousal_changed(self, val):
        """Update display label when arousal slider changes."""
        try:
            val_float = float(val)
            self.arousal_var.set(val_float)
            self.arousal_value_label.config(text=f"{val_float:.2f}")
        except AttributeError as e:
            print(f"[ERROR] AttributeError in on_arousal_changed: {e}")
        except Exception as e:
            print(f"[ERROR] Unexpected error in on_arousal_changed: {e}")

    def on_generate_clicked(self):
        """Gather parameters and start generation in a thread."""
        self.logs_text.delete('1.0', tk.END)
        self.errors_label_var.set("")
        self.status_label_var.set("Starting generation...")
        self.progress_bar['value'] = 0

        # Build args from the current UI state
        args = {
            "use_ai": self.use_ai_var.get(),
            "prompt": self.prompt_var.get().strip(),
            "key": self.key_var.get().strip(),
            "num_bars": self.num_bars_var.get(),
            "gen_len": self.gen_len_var.get(),
            "max_input_len": self.max_input_len_var.get(),
            "temp_notes": float(self.temp_notes_var.get()),
            "temp_rests": float(self.temp_rests_var.get()),
            "temp": [float(self.temp_notes_var.get()), float(self.temp_rests_var.get())],
            "topk": self.topk_var.get(),
            "topp": float(self.topp_var.get()),
            "penalty_coeff": float(self.penalty_var.get()),
            "no_amp": self.no_amp_var.get(),
            "prune": self.prune_var.get(),
            "no_cuda": self.no_cuda_var.get(),
            "counter_melody_hyperness": float(self.counter_slider.get()),
            "out_dir": self.generator.output,
            "model_dir": get_data_path("models/continuous_concat")
        }

        # If AI is not used, we also capture valence and arousal
        if not args["use_ai"]:
            args["valence"] = [float(self.valence_slider.get())]
            args["arousal"] = [float(self.arousal_slider.get())]

        # Validate inputs
        is_valid, missing_fields = self.generator.validate_inputs(args)
        if not is_valid:
            missing_list = ", ".join(missing_fields)
            messagebox.showwarning("Validation Error", f"Missing required fields: {missing_list}")
            return

        # Ensure output directory is set
        if not self.generator.output:
            messagebox.showwarning("Output Not Set", "Please set an output directory first.")
            return

        # Start background thread
        self.progress_queue = Queue()
        self.thread = threading.Thread(target=self.generator.generate_midi, args=(args, self.progress_queue))
        self.thread.start()

        # Start polling the queue
        self.after(self.poll_interval_ms, self.poll_queue)

    def poll_queue(self):
        """Poll the progress_queue for updates and update the UI."""
        # Read messages in the queue
        while self.progress_queue and not self.progress_queue.empty():
            msg = self.progress_queue.get()
            if not msg:
                continue

            msg_type, *rest = msg
            if msg_type == "progress":
                progress_val, status_str = rest
                pct = int(progress_val * 100)
                self.progress_bar['value'] = pct
                self.status_label_var.set(status_str)

            elif msg_type == "log":
                line = rest[0]
                self.logs_text.insert(tk.END, line + "\n")
                self.logs_text.see(tk.END)

            elif msg_type == "info":
                info_line = rest[0]
                self.logs_text.insert(tk.END, f"[INFO] {info_line}\n")
                self.logs_text.see(tk.END)

            elif msg_type == "error":
                err_line = rest[0]
                self.logs_text.insert(tk.END, f"[ERROR] {err_line}\n")
                self.logs_text.see(tk.END)
                self.errors_label_var.set(err_line)

            elif msg_type == "success":
                success_msg = rest[0]
                self.logs_text.insert(tk.END, f"[SUCCESS] {success_msg}\n")
                self.logs_text.see(tk.END)
                self.status_label_var.set("Done")
                self.progress_bar['value'] = 100

        # If the thread is still running, continue polling
        if self.thread and self.thread.is_alive():
            self.after(self.poll_interval_ms, self.poll_queue)
        else:
            # Final check to process any remaining messages
            if self.progress_queue and not self.progress_queue.empty():
                self.after(self.poll_interval_ms, self.poll_queue)
            else:
                self.status_label_var.set("Generation complete!")
                self.progress_bar['value'] = 100

def main():
    app = MainApplication()
    app.mainloop()
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Unhandled exception occurred:")
        traceback.print_exc()
