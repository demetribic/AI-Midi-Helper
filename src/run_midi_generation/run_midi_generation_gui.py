import streamlit as st
import subprocess
import sys
import os
import threading
from queue import Queue
import stat
import time
from pathlib import Path
import re

class MIDI_Generator:
    def __init__(self):
        def install_requirements(requirements_file):
            """
            Install requirements from a given requirements.txt file.

            Args:
                requirements_file (str): Path to the requirements.txt file.
            """
            print(f"Installing requirements from {os.path.abspath(requirements_file)}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
                print(f"All requirements from {requirements_file} installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error during installation: {e}")
                sys.exit(1)
        install_requirements("requirements.txt")
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
            "no_cuda": {"type": "checkbox", "default": True, "label": "Use CPU (No CUDA/MPS - reccomended for macOS)"},
            "counter_melody_hyperness": {"type": "number", "default": 0.5, "label": "Counter Melody Hyperness", "required": False}
                    }
        self.log_path = os.path.abspath("output/logs/generation.log")  # Adjust this path as needed
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def validate_inputs(self, args):
        """Validate all input parameters based on AI assistance."""
        required_fields = [k for k, v in self.parameters.items() if v.get("required", False)]
        
        if args.get("use_ai"):
            # When AI assistance is enabled, 'prompt' is required
            fields_to_check = required_fields
        else:
            # When AI assistance is disabled, 'valence' and 'arousal' are required
            fields_to_check = required_fields + ['valence', 'arousal']
        
        for field in fields_to_check:
            if field not in args or args.get(field) in [None, '']:
                label = self.parameters.get(field, {}).get("label", field.capitalize())
                st.error(f"'{label}' is required.")
                return False
        return True

    def set_permissions(self, path):
        """Ensure the directory has the correct permissions."""
        try:
            os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
        except Exception as e:
            raise PermissionError(f"Failed to set permissions for {path}: {e}")

    def monitor_log_progress(self, progress_queue):
        token_pattern = re.compile(r"Generating token (\d+)/(\d+)")
        last_position = 0

        while True:
            try:
                with open(self.log_path, 'r') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()

                    if new_lines:  # Only process if there are new lines
                        for line in new_lines:
                            token_match = token_pattern.search(line)
                            if token_match:
                                current_token = int(token_match.group(1))
                                total_tokens = int(token_match.group(2))
                                progress = current_token / total_tokens
                                progress_queue.put(("progress", progress, f"Generating token {current_token}/{total_tokens}"))

                            # Send all log lines to logs
                            progress_queue.put(("log", line.strip()))

                    # Sleep to reduce CPU usage
                    time.sleep(0.1)

            except Exception as e:
                progress_queue.put(("error", f"Error reading log file: {str(e)}"))
                break

    def generate_midi(self, args, progress_queue):
        try:
            # Start log monitoring in a separate thread
            log_monitor = threading.Thread(
                target=self.monitor_log_progress,
                args=(progress_queue,)
            )
            log_monitor.daemon = True  # This ensures the thread stops when the main thread stops
            log_monitor.start()

            # Define directories
            output_dir = os.path.abspath("output/generated_midis")
            from datetime import datetime
            processed_dir = os.path.abspath("output/processed_midis")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            processed_dir = os.path.abspath(f"output/processed_midis/{timestamp}/")
            os.makedirs(processed_dir, exist_ok=True)
            model_dir = os.path.abspath(f"output/models/continuous_concat")
            os.makedirs(output_dir, exist_ok=True)

            if not os.path.isdir(model_dir):
                progress_queue.put(("error", f"Model directory not found: {model_dir}"))
                return

            # Build generation command
            generate_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                 "generate_command.py"
            )

            if not os.path.exists(generate_script):
                progress_queue.put(("error", f"generate_command.py not found at {generate_script}"))
                return

            command = [
                sys.executable, generate_script,
                "--conditioning", "continuous_concat",
                "--model_dir", model_dir,
                "--out_dir", output_dir,
                "--gen_len", str(args["gen_len"]),
                "--max_input_len", str(args["max_input_len"]),
                "--temp", str(args["temp_notes"]), str(args["temp_rests"]),
                "--topk", str(args["topk"]),
                "--topp", str(args["topp"]),
                "--penalty_coeff", str(args["penalty_coeff"]),
            ]

            for flag in ["no_cuda", "no_amp", "quiet", "short_filename"]:
                if args.get(flag):
                    command.append(f"--{flag}")

            # Include prompt if AI assistance is enabled
            if args.get("use_ai"):
                command.extend(["--prompt", args["prompt"]])
            else:
                # Include valence and arousal if AI assistance is disabled
                command.extend(["--valence", str(args["valence"]), "--arousal", str(args["arousal"])])

            progress_queue.put(("info", f"Executing command: {' '.join(command)}"))

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Stream output
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    if "Device set to use" in output:
                        progress_queue.put(("info", output.strip()))
                    else:
                        progress_queue.put(("log", output.strip()))

            stderr_output = process.stderr.read()
            if stderr_output:
                if "emoji" in stderr_output:  # Specific handling for missing emoji library
                    progress_queue.put(("error", "Emoji library is missing. Install it using: pip install emoji==0.6.0"))
                else:
                    progress_queue.put(("error", stderr_output.strip()))

            if process.returncode != 0:
                progress_queue.put(("error", f"Generation failed with return code {process.returncode}"))
                return

            self.post_process(output_dir, processed_dir, progress_queue, args['key'], args['num_bars'], float(args['counter_melody_hyperness']))

        except Exception as e:
            progress_queue.put(("error", f"Error during generation: {str(e)}"))
    def post_process(self, input_dir, output_dir, progress_queue, key, num_bars, counter_melody_hyperness):
        """Run post-processing on generated MIDI."""
        try:
            # Find latest MIDI file
            midi_files = [f for f in os.listdir(input_dir) if f.endswith('.mid')]
            if not midi_files:
                progress_queue.put(("error", f"No MIDI files found in {input_dir}"))
                return

            # Get the full path of the latest MIDI file
            latest_midi = max(
                [os.path.join(input_dir, f) for f in midi_files],
                key=os.path.getmtime
            )

            progress_queue.put(("info", f"Processing: {latest_midi}"))

            # Run post-processing
            post_process_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "post_process.py"
            )

            if not os.path.exists(post_process_script):
                progress_queue.put(("error", f"post_process.py not found at {post_process_script}"))
                return

            command = [
                sys.executable,
                post_process_script,
                "--input", latest_midi,
                "--output_dir", output_dir,
                "--num_bars", str(num_bars),
                "--counter_hyperness", str(counter_melody_hyperness),
                "--key", key
            ]

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

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

def main():
    st.set_page_config(
        page_title="AI MIDI Generation Interface",
        page_icon="🎵",
        layout="wide"
    )

    st.title("🎵 AI MIDI Generation Interface")

    generator = MIDI_Generator()

    # Add log path configuration in the UI
    with st.expander("Advanced Settings"):
        custom_log_path = st.text_input(
            "Log File Path",
            value=str(generator.log_path),
            help="Path to the MIDI generation log file"
        )
        generator.log_path = Path(custom_log_path)

    # AI assistance checkbox
    use_ai = st.checkbox("Let AI Control Parameters Valence, Arousal, & Temps Based on Prompt", value=False)

    # Create two columns for parameters
    col1, col2 = st.columns(2)

    # Initialize parameters dictionary
    args = {}

    # Split parameters between columns
    params_list = list(generator.parameters.items())
    mid_point = len(params_list) // 2


    with col1:
        for key, param in params_list[:mid_point]:
            if param["type"] == "text":
                if key == "prompt":
                    args[key] = st.text_input(
                        param["label"],
                        value=param["default"],
                        disabled=not use_ai
                    )
                else:
                    args[key] = st.text_input(param["label"], value=param["default"])
            elif param["type"] == "select":
                args[key] = st.selectbox(
                    param["label"],
                    param["options"],
                    index=param["options"].index(param["default"])
                )
            elif param["type"] == "number":
                args[key] = st.number_input(param["label"], value=param["default"])
            elif param["type"] == "checkbox":
                args[key] = st.checkbox(param["label"], value=param["default"])

    with col2:
        for key, param in params_list[mid_point:]:
            if param["type"] == "text":
                args[key] = st.text_input(param["label"], value=param["default"])
            elif param["type"] == "select":
                args[key] = st.selectbox(
                    param["label"],
                    param["options"],
                    index=param["options"].index(param["default"])
                )
            elif param["type"] == "number":
                args[key] = st.number_input(param["label"], value=param["default"])
            elif param["type"] == "checkbox":
                args[key] = st.checkbox(param["label"], value=param["default"])

    # Add Counter Melody Hyperness Slider
    counter_melody_hyperness = st.slider(
        "Counter Melody Hyperness",
        min_value=0.0,
        max_value=1.0,
        value=generator.parameters["counter_melody_hyperness"]["default"],
        step=0.01,
        help="Adjusts the hyperness of the counter melody (0.0 = calm, 1.0 = hyper)."
    )
    args["counter_melody_hyperness"] = counter_melody_hyperness

    # Conditional UI based on AI assistance
    if use_ai:
        st.info("Temperatures may change based on AI control.")
    else:
        st.subheader("User-Controlled Parameters")
        valence = st.slider(
            "Valence",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
            help="A measure of the musical emotion (0.0 = negative, 1.0 = positive - anything more can be slightly excessive)."
        )
        arousal = st.slider(
            "Arousal",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.01,
            help="A measure of the musical energy (0.0 = calm, 1.0 = positive - anything more can be excessive)."
        )
        args["valence"] = valence
        args["arousal"] = arousal
        # Shade out the prompt if AI is off
        if "prompt" in args:
            args["prompt"] = args.get("prompt", "")
    # Progress area
    progress_placeholder = st.empty()
    log_placeholder = st.empty()
    error_placeholder = st.empty()
    if st.button("Generate MIDI"):
        if not generator.validate_inputs(args):
            st.stop()

        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        log_container = st.container()
        error_placeholder = st.empty()

        progress_queue = Queue()

        # Start generation in separate thread
        thread = threading.Thread(
            target=generator.generate_midi,
            args=(args, progress_queue)
        )
        thread.start()

        # Output logs and errors
        log_output = []  # Store logs as a list of lines

        while thread.is_alive() or not progress_queue.empty():
            try:
                msg = progress_queue.get(timeout=0.1)
                msg_type, *message = msg
                if msg_type == "progress":
                    progress_value, status = message
                    progress_bar.progress(progress_value)
                    status_text.text(status)
                elif msg_type == "error":
                    error_placeholder.error(message[0])
                elif msg_type == "success":
                    st.success(message[0])
                elif msg_type == "info":
                    log_output.append(f"INFO: {message[0]}")
                    log_placeholder.text_area("Info Logs", "\n".join(log_output), height=150)
                elif msg_type == "log":
                    log_output.append(message[0])
                    log_placeholder.text_area("Logs", "\n".join(log_output), height=300)
            except:
                pass

        progress_bar.progress(1.0)
        status_text.text("Generation complete!")
        thread.join()

if __name__ == "__main__":
    main()
