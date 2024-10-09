# MarianMT to CoreML Converter

This project provides a Python script to convert Hugging Face's MarianMT translation models to Apple's CoreML format. This enables the use of these powerful translation models in iOS and macOS applications.

## Project Structure

```
MarianMT-to-CoreML-Converter/
│
├── src/
│   └── marianmt_to_coreml.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Prerequisites

- Python 3.9 (specifically 3.9.13 recommended)
- pip (Python package installer)
- pyenv (Python version management tool)
- Git

## Setup

1. Install pyenv if you haven't already:
   
   On macOS (using Homebrew):
   ```
   brew update
   brew install pyenv
   ```

   On Linux:
   ```
   curl https://pyenv.run | bash
   ```

2. Add pyenv to your shell:

   Add these lines to your `.bashrc`, `.zshrc`, or whatever shell configuration file you use:
   ```
   export PYENV_ROOT="$HOME/.pyenv"
   export PATH="$PYENV_ROOT/bin:$PATH"
   eval "$(pyenv init --path)"
   eval "$(pyenv init -)"
   ```

   Then restart your shell or run:
   ```
   source ~/.bashrc  # or ~/.zshrc, depending on your shell
   ```

3. Install Python 3.9.13 using pyenv:
   ```
   pyenv install 3.9.13
   ```

4. Clone the repository:
   ```
   git clone https://github.com/yourusername/MarianMT-to-CoreML-Converter.git
   cd MarianMT-to-CoreML-Converter
   ```

5. Set the local Python version and create a virtual environment:
   ```
   pyenv local 3.9.13
   python -m venv venv
   source venv/bin/activate
   ```

6. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

The script can be used to convert any MarianMT model to CoreML format. Here's how to use it:

```
python src/marianmt_to_coreml.py --model MODEL_NAME --example 'Example text' --output OUTPUT_FILE.mlmodel
```

Arguments:
- `--model`: The name or path of the MarianMT model (required)
- `--tokenizer`: The name or path of the tokenizer, if different from the model (optional)
- `--example`: An example text for tokenization (required)
- `--output`: The output filename for the CoreML model (required)

Note: Use single quotes ('') around the example text to avoid issues with shell interpretation of special characters.

Examples:

1. Convert Chinese to English model:
   ```
   python src/marianmt_to_coreml.py --model Helsinki-NLP/opus-mt-zh-en --example '你好世界' --output opus-mt-zh-en.mlmodel
   ```

2. Convert English to Chinese model:
   ```
   python src/marianmt_to_coreml.py --model Helsinki-NLP/opus-mt-en-zh --example 'Hello, world!' --output opus-mt-en-zh.mlmodel
   ```

3. Convert French to English model:
   ```
   python src/marianmt_to_coreml.py --model Helsinki-NLP/opus-mt-fr-en --example 'Bonjour le monde!' --output opus-mt-fr-en.mlmodel
   ```

The script will generate a `conversion.log` file with detailed information about the conversion process.

## Troubleshooting

- If you encounter memory issues, try reducing the `max_length` parameter in the tokenizer call within the `convert_marianmt_to_coreml` function.
- Ensure you have the correct versions of `torch`, `transformers`, and `coremltools` installed as specified in the `requirements.txt` file.
- Check the `conversion.log` file for detailed error messages and the conversion process log.
- If you're having issues with Python versions, make sure you've followed the pyenv setup instructions correctly and that you're using the virtual environment.

### Quoting Issues

If you encounter issues with quoting, try the following:

- Use single quotes ('') instead of double quotes ("") around the example text.
- If single quotes don't work, try escaping the quotes: \"Example text\"
- If you're still having issues, you can split the command across multiple lines using the backslash (\) character:

  ```
  python src/marianmt_to_coreml.py \
  --model Helsinki-NLP/opus-mt-fr-en \
  --example 'Bonjour le monde!' \
  --output opus-mt-fr-en.mlmodel
  ```

## Contributing

Contributions to improve the script or extend its functionality are welcome. Please feel free to submit a pull request or open an issue for discussion.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for providing the MarianMT models
- Apple for the CoreML framework
- The open-source community for the various libraries used in this project

## Note on Python Version

This project is specifically designed to work with Python 3.9.13. Using other versions of Python may lead to compatibility issues with the required libraries. If you encounter any version-related problems, please ensure you're using Python 3.9.13 as specified in the setup instructions.
