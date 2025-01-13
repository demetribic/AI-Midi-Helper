import json
import subprocess
import shlex
import os
import sys
import torch
import traceback
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir
while not (project_root / 'models').exists():
    if project_root == project_root.parent:
        raise FileNotFoundError("Could not find project root (setup.py)")
    project_root = project_root.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
from midi_app.generate_utils import validate_output_dir
from midi_app.generate import generate_midi

def log_exception(exc_type, exc_value, exc_traceback):
    """
    Custom exception hook to log uncaught exceptions.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Call the default exception handler for KeyboardInterrupt
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    import traceback
    print("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    # Optionally, print the traceback to the console
    traceback.print_exception(exc_type, exc_value, exc_traceback)



# Configure print and other constants
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def load_sentiment_pipeline(cache_directory):
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
        print(f"Failed to load sentiment pipeline: {e}")
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
        print(f"Failed to parse user input: {e}")
        sys.exit(1)


def adjust_parameters_based_on_prompt(args, valence, arousal):
    # Adjust generation temperature with capping
    args.temp[0] = min(max(args.temp[0] * (1 + arousal), 0.5), 2.5)  # Notes
    args.temp[1] = min(max(args.temp[1] * (1 + arousal), 0.5), 2.5)  # Rests

    # Adjust top-p sampling
    if arousal > 0.5:
        args.topp = min(1.0, args.topp + 0.1)  # Allow more diversity with higher arousal

    # print updated parameters for debugging
    print(f"[Adjusted Parameters] gen_len: {args.gen_len}, temp: {args.temp}, topp: {args.topp}")
    return args


def generate_command(args_dict):
    """
    Executes the MIDI generation process by calling generate_midi directly.

    Parameters:
        args_dict (dict): Dictionary containing generation parameters
    """
    # Convert dictionary args to an object for compatibility
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    # Set default values for missing arguments
    default_args = {
        'model_dir': None,
        'num_runs': 1,
        'debug': False,
        'seed': None,
        'conditioning': None,
        'quiet': False,
        'short_filename': False,
        'batch_size': 1,
        'min_n_instruments': 1,
        'batch_gen_dir': None,
        'arousal_feature': None,
        'max_density': None,
        'window_size': None,
        'chord_threshold': None,
        'valence': [0.5],  # Default valence 
        'arousal_val': [0.5],  # Default arousal 
    }

    # Combine default args with provided args
    combined_args = {**default_args, **args_dict}
    args = Args(**combined_args)

    # Validate output directory
    if not hasattr(args, 'out_dir') or not args.out_dir:
        raise ValueError("Output directory not specified")
    
    validate_output_dir(args.out_dir)

    # Validate model directory
    if not os.path.exists(args.model_dir):
        raise ValueError(f"Model directory not found: {args.model_dir}")

    if args.use_ai:
        try:
            classifier = load_sentiment_pipeline()
            params = parse_user_input(args.prompt, classifier)
            valence, arousal, sentiment = params['valence'], params['arousal'], params['sentiment']
            print(f"[NLP] Sentiment: {sentiment.capitalize()}, Valence: {valence}, Arousal: {arousal}")
            args = adjust_parameters_based_on_prompt(args, valence, arousal)
        except Exception as e:
            print(f"Failed to process AI parameters: {e}")
            raise

    else:
        valence, arousal = args.valence[0], args.arousal_val[0]

    try:
        # Setup print for generation
        print(f"Output directory: {args.out_dir}")
        print(f"Model directory: {args.model_dir}")
        
        # Call generate_midi directly
        generate_midi(
            model_dir=args.model_dir,
            cpu=args.no_cuda,
            num_runs=args.num_runs,
            gen_len=args.gen_len,
            max_input_len=args.max_input_len,
            temp=args.temp,
            topk=args.topk,
            topp=args.topp,
            debug=args.debug,
            seed=args.seed,
            no_amp=args.no_amp,
            conditioning=args.conditioning,
            penalty_coeff=args.penalty_coeff,
            quiet=args.quiet,
            short_filename=args.short_filename,
            batch_size=args.batch_size,
            min_n_instruments=args.min_n_instruments,
            batch_gen_dir=args.batch_gen_dir,
            out_dir=args.out_dir,
            arousal_feature=args.arousal_feature,
            valence=[valence],
            arousal_val=[arousal],
            max_density=args.max_density,
            window_size=args.window_size,
            chord_threshold=args.chord_threshold,
            prune=args.prune
        )
        
        print("[Success] MIDI generation completed.")
    except Exception as e:
        print(f"Failed during MIDI generation: {e}")
        raise

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate MIDI based on user input.')

    # Required arguments
    parser.add_argument('--prompt', type=str, help='User description for MIDI generation.')
    parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
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
    args = parser.parse_args()
    try:
        generate_command(args)
    except Exception as e:
        print(f"[Error] Failed to construct command: {e}")
        sys.exit(1)
    


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
