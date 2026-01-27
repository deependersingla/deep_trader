import argparse
import subprocess
import os
import sys

def run_script(path):
    print(f"--- Launching {path} ---")
    # Use the same python interpreter
    python_exe = sys.executable
    try:
        subprocess.check_call([python_exe, path])
    except subprocess.CalledProcessError as e:
        print(f"Error running {path}: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

def main():
    parser = argparse.ArgumentParser(description="Deep Trader Modern CLI")
    parser.add_argument('mode', choices=['dqn', 'pg', 'supervised', 'analog', 'ib'], 
                        help="Mode to run: dqn (RL), pg (Policy Gradient), supervised (Classif), analog (Regression), ib (Data Download)")
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.mode == 'dqn':
        script = os.path.join(base_dir, 'tensor-reinforcement', 'train_modern.py')
        run_script(script)
        
    elif args.mode == 'pg':
        script = os.path.join(base_dir, 'tensor-reinforcement', 'train_pg_modern.py')
        run_script(script)
        
    elif args.mode == 'supervised':
        script = os.path.join(base_dir, 'supervised', 'modern_train.py')
        run_script(script)
        
    elif args.mode == 'analog':
        script = os.path.join(base_dir, 'supervised', 'modern_train_analog.py')
        run_script(script)
        
    elif args.mode == 'ib':
        script = os.path.join(base_dir, 'ib', 'modern_ib_client.py')
        run_script(script)

if __name__ == "__main__":
    main()
