import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import coremltools as ct
from transformers import MarianMTModel, MarianTokenizer
import traceback
import sys
import logging
import argparse

# Set up logging
logging.basicConfig(filename='conversion.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]])
        return self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids).logits

def convert_marianmt_to_coreml(model_name, tokenizer_name, example_text, output_filename):
    try:
        logging.info(f"Starting conversion process for {model_name}")
        logging.info(f"Python version: {sys.version}")
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"Coremltools version: {ct.__version__}")

        logging.info(f"Loading model and tokenizer for {model_name}...")
        tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)
        model = MarianMTModel.from_pretrained(model_name).eval()
        wrapped_model = ModelWrapper(model)
        logging.info("Model and tokenizer loaded successfully.")

        logging.info(f"Tokenizing example input text: '{example_text}'")
        inputs = tokenizer(example_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        logging.info("Tokenization completed.")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        logging.info(f"Tracing the MarianMT model '{model_name}' with TorchScript...")
        wrapped_model.eval()

        with torch.no_grad():
            traced_model = torch.jit.trace(wrapped_model, (input_ids, attention_mask))
            torch.jit.save(traced_model, "traced_model.pt")
        logging.info("Tracing completed successfully.")

        logging.info("Loading traced model...")
        traced_model = torch.jit.load("traced_model.pt")
        logging.info("Traced model loaded successfully.")

        logging.info("Starting CoreML conversion...")
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(shape=input_ids.shape, dtype=int, name="input_ids"),
                ct.TensorType(shape=attention_mask.shape, dtype=int, name="attention_mask"),
            ],
            outputs=[ct.TensorType(name="logits")],
            minimum_deployment_target=ct.target.iOS14
        )
        logging.info("CoreML conversion completed successfully.")

        logging.info(f"Saving CoreML model to '{output_filename}'...")
        mlmodel.save(output_filename)
        logging.info(f"CoreML model saved successfully to '{output_filename}'\n")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error("Traceback:")
        logging.error(traceback.format_exc())
        raise

def main():
    parser = argparse.ArgumentParser(description="Convert MarianMT models to CoreML format.")
    parser.add_argument("--model", required=True, help="Name or path of the MarianMT model")
    parser.add_argument("--tokenizer", help="Name or path of the tokenizer (if different from model)")
    parser.add_argument("--example", required=True, help="Example text for tokenization")
    parser.add_argument("--output", required=True, help="Output filename for the CoreML model")

    args = parser.parse_args()

    tokenizer_name = args.tokenizer if args.tokenizer else args.model

    logging.info("Script starting...")
    try:
        convert_marianmt_to_coreml(args.model, tokenizer_name, args.example, args.output)
        logging.info("Conversion completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred in main: {str(e)}")
        logging.error("Traceback:")
        logging.error(traceback.format_exc())
    logging.info("Script finished.")

if __name__ == "__main__":
    main()
