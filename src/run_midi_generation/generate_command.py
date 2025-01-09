import json
import subprocess
import shlex
import os
import sys
import torch
import traceback
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Configure logging
log_filename = "output/logs/generation_output.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # 'w' to overwrite the log file each time, 'a' to append
)

# Redirect stdout and stderr to the log file
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.strip():  # Ignore empty messages
            if "emoji is not installed" in message or "set to" in message:
                logging.info(message)  # Treat as informational
            else:
                self.level(message)

    def flush(self):  # Needed for compatibility with `sys.stdout`/`sys.stderr`
        pass

sys.stdout = LoggerWriter(logging.info)
sys.stderr = LoggerWriter(logging.error)

def validate_midi_generation_args(args):
    """
    Validates the arguments for MIDI generation and provides helpful error messages.
    
    Parameters:
        args (argparse.Namespace): The parsed command line arguments
        
    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
        
    if not args.conditioning:
        return False, "Conditioning type must be specified (none, discrete_token, continuous_token, continuous_concat)"
    
    if not args.model_dir or not os.path.exists(args.model_dir):
        return False, f"Model directory '{args.model_dir}' does not exist"
        
    if args.gen_len < 1:
        return False, "Generation length must be positive"
        
    if args.max_input_len < 1:
        return False, "Max input length must be positive"
        
    if len(args.temp) != 2 or any(t <= 0 for t in args.temp):
        return False, "Temperature must be two positive values"
        
    if args.batch_size < 1:
        return False, "Batch size must be positive"
        
    if args.min_n_instruments < 1:
        return False, "Minimum number of instruments must be positive"
        
    return True, ""

def format_command(args):
    """
    Formats the command with proper escaping and validation.
    
    Parameters:
        args (argparse.Namespace): The parsed command line arguments
        
    Returns:
        str: Properly formatted command
    """
    base_cmd = [
        "python3", "src/run_midi_generation/generate_command.py",
        "--prompt", f'"{args.prompt}"',
        "--conditioning", args.conditioning,
    ]
    
    if args.model_dir:
        base_cmd.extend(["--model_dir", args.model_dir])
        

        
    optional_flags = [
        ("--gen_len", args.gen_len),
        ("--max_input_len", args.max_input_len),
        ("--temp", f"{args.temp[0]} {args.temp[1]}"),
        ("--topk", args.topk),
        ("--topp", args.topp),
        ("--seed", args.seed),
        ("--penalty_coeff", args.penalty_coeff),
        ("--batch_size", args.batch_size),
        ("--min_n_instruments", args.min_n_instruments),
    ]
    
    for flag, value in optional_flags:
        if value is not None:
            base_cmd.extend([flag, str(value)])
            
    if args.no_cuda:
        base_cmd.append("--cpu")
        
    return " ".join(base_cmd)

def load_sentiment_pipeline(cache_directory='output/models/'):
    try:
        cache_directory = os.path.abspath(cache_directory)
        print(f"caching directory: {cache_directory}")
        os.makedirs(cache_directory, exist_ok=True)
        model_name = "finiteautomata/bertweet-base-sentiment-analysis"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=cache_directory
        )
        print("model loaded")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_directory
        )
        print("tokenizer loaded")
        classifier = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=-1  # Force CPU usage
        )
        print("pipeline loaded")
        return classifier
    except Exception as e:
        logging.error(f"Failed to load sentiment pipeline: {e}")
        sys.exit(1)


def parse_user_input(prompt, classifier):
    """
    Uses a local sentiment classifier to compute valence and arousal from the user prompt.
    
    Parameters:
        prompt (str): User-provided description for MIDI generation.
        classifier: Hugging Face sentiment analysis pipeline.
    
    Returns:
        dict: A dictionary containing valence, arousal, and detected sentiment.
    """
    try:
        print("calling classifier")
        results = classifier(prompt)
        print("classifier called")
        if not results:
            return {"valence": 0.0, "arousal": 0.0, "sentiment": "neutral"}
        
        sentiment = results[0]['label']
        score = results[0]['score']
        
        valence = score if sentiment == "POSITIVE" else -score if sentiment == "NEGATIVE" else 0.0
        arousal = min(score, 0.8)
        valence = min(valence, 0.8)
        return {"valence": valence, "arousal": arousal, "sentiment": sentiment.lower()}
    except Exception as e:
        logging.error(f"Failed to parse user input: {e}")
        sys.exit(1)


def adjust_parameters_based_on_prompt(args, valence, arousal):
    # Adjust generation temperature with capping
    args.temp[0] = min(max(args.temp[0] * (1 + arousal), 0.5), 2.5)  # Notes
    args.temp[1] = min(max(args.temp[1] * (1 + arousal), 0.5), 2.5)  # Rests

    # Adjust top-p sampling
    if arousal > 0.5:
        args.topp = min(1.0, args.topp + 0.1)  # Allow more diversity with higher arousal

    # Logging updated parameters for debugging
    logging.info(f"[Adjusted Parameters] gen_len: {args.gen_len}, temp: {args.temp}, topp: {args.topp}")
    return args


def generate_command(args, valence, arousal):
    """
    Constructs the command to run generate.py based on extracted parameters.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(dir_path, 'generate.py')
    # Print the script path to debug
    print(f"Looking for generate.py at: {script_path}")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"generate.py not found at {script_path}")
    command = [
        sys.executable, script_path,
        "--model_dir", args.model_dir,
        "--conditioning", args.conditioning,
        "--valence", str(valence),
        "--arousal_val", str(arousal),
        "--gen_len", str(args.gen_len),
        "--batch_size", str(args.batch_size),
        "--min_n_instruments", str(args.min_n_instruments),
        "--out_dir", args.out_dir
    ]

    if args.no_cuda:
        command.append("--cpu")
    if args.debug:
        command.append("--debug")
    if args.no_amp:
        command.append("--no_amp")
    if args.quiet:
        command.append("--quiet")
    if args.short_filename:
        command.append("--short_filename")
    if args.batch_gen_dir:
        command.extend(["--batch_gen_dir", args.batch_gen_dir])

    command.extend([
        "--num_runs", str(args.num_runs),
        "--max_input_len", str(args.max_input_len),
        "--temp", str(args.temp[0]), str(args.temp[1]),
        "--topk", str(args.topk),
        "--topp", str(args.topp),
        "--seed", str(args.seed),
        "--penalty_coeff", str(args.penalty_coeff)
    ])
    print("command: ", command)
    return command


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate MIDI based on user input.')

    # Required arguments
    parser.add_argument('--prompt', type=str, help='User description for MIDI generation.')
    parser.add_argument('--model_dir', type=str, default='', help='Model directory')
    
    parser.add_argument('--conditioning', type=str, default='continuous_concat',
                        choices=["none", "discrete_token", "continuous_token", "continuous_concat"],
                        help='Conditioning type')

    # Optional arguments
    parser.add_argument('--no_cuda', action='store_true', help="Use CPU instead of GPU")
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--gen_len', type=int, default=400, help='Max generation length in tokens')
    parser.add_argument('--max_input_len', type=int, default=400, help='Max input length in tokens')
    parser.add_argument('--temp', type=float, nargs=2, default=[1.2, 1.2], help='Generation temperature (notes, rests)')
    parser.add_argument('--topk', type=int, default=-1, help='Top-k sampling')
    parser.add_argument('--topp', type=float, default=0.7, help='Top-p (nucleus) sampling')
    parser.add_argument('--debug', action='store_true', help="Run in debug mode (don't save files)")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--no_amp', action='store_true', help="Disable automatic mixed precision")
    parser.add_argument('--penalty_coeff', type=float, default=0.5, help="Coefficient for penalizing repeating notes")
    parser.add_argument('--quiet', action='store_true', help="Suppress verbose output")
    parser.add_argument('--short_filename', action='store_true', help="Use short filenames for output")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--min_n_instruments', type=int, default=1, help='Minimum number of instruments required')
    parser.add_argument('--batch_gen_dir', type=str, default="", help="Subdirectory name for batch generation")
    parser.add_argument('--out_dir', type=str, default="", help="Subdirectory name for batch generation")

    # Renamed the continuous arousal argument to avoid conflict
    parser.add_argument('--valence', type=float, nargs='+', default=[0.8],
                        help='Continuous valence value(s) for conditioning')
    parser.add_argument('--arousal_val', type=float, nargs='+', default=[0.8],
                        help='Continuous arousal value(s) for conditioning')
    
    # Add the --use_ai flag
    parser.add_argument('--use_ai', action='store_true', help="Enable AI-assisted MIDI generation")

    # Parse arguments
    args = parser.parse_args()

    # Conditional validation for --prompt
    if args.use_ai and not args.prompt:
        parser.error("--prompt is required when --use_ai is set.")
    
    # Continue with the rest of your program
    print("Arguments parsed successfully!")
    print(f"Use AI: {args.use_ai}, Prompt: {args.prompt}")

    if args.model_dir == '':
        args.model_dir = os.path.abspath(f"output/models/{args.conditioning}")
    if args.out_dir == '':
        args.out_dir = os.path.abspath(f"output/generated_midis/")
    
    is_valid, error_message = validate_midi_generation_args(args)
    if not is_valid:
        logging.error(f"Invalid arguments: {error_message}")
        sys.exit(1)


    # Print all arguments for debugging
    print("Received arguments:", vars(args))
    
    # Validate model directory
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)
        
    # Validate output directory
    try:
        os.makedirs(args.out_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory: {e}")
        sys.exit(1)
    if args.use_ai:
        try:
            classifier = load_sentiment_pipeline()
            print("classifier loaded")
        except Exception as e:
            logging.info(f"[Error] Failed to load sentiment analysis pipeline: {e}")
            sys.exit(1)

        try:
            print("parsing user input")
            params = parse_user_input(args.prompt, classifier)
            print("user input parsed")
            valence, arousal, sentiment = params['valence'], params['arousal'], params['sentiment']
        except Exception as e:
            logging.info(f"[Error] Failed to parse user input: {e}")
            sys.exit(1)

        logging.info(f"[NLP] Sentiment: {sentiment.capitalize()}, Valence: {valence}, Arousal: {arousal}")
        args = adjust_parameters_based_on_prompt(args, valence, arousal)
    else:
        valence, arousal = args.valence[0], args.arousal_val[0]
    try:
        command = generate_command(args, valence, arousal)
        logging.info(f"[Command] {' '.join(shlex.quote(arg) for arg in command)}")
    except Exception as e:
        logging.info(f"[Error] Failed to construct command: {e}")
        sys.exit(1)
    try:
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        ) as process:
            # Read stdout and stderr concurrently
            stdout, stderr = process.communicate()

            # Print stdout in real-time
            for line in stdout.splitlines():
                print(line)

            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, command, output=stdout, stderr=stderr
                )

        logging.info("[Success] MIDI generation completed.")
    except subprocess.CalledProcessError as e:
        logging.error(f"[Error] MIDI generation failed:\n{e.stderr}")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)
