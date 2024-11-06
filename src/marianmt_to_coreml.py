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
logging.basicConfig(
    filename='conversion.log',
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]])
        return self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids).logits

def convert_marianmt_to_coreml(model_name, tokenizer_name, example_text, output_filename):
    try:
        logging.info("Starting conversion process.")
        logging.info(f"Model: {model_name}, Tokenizer: {tokenizer_name}")
        
        logging.info("Loading model and tokenizer...")
        tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)
        model = MarianMTModel.from_pretrained(model_name).eval()
        wrapped_model = ModelWrapper(model)
        logging.info("Model and tokenizer loaded successfully.")

        logging.info("Tokenizing example input text.")
        inputs = tokenizer(example_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        logging.info("Tokenization completed.")

        logging.info("Tracing the MarianMT model with TorchScript.")
        with torch.no_grad():
            traced_model = torch.jit.trace(wrapped_model, (input_ids, attention_mask))
            torch.jit.save(traced_model, "traced_model.pt")
        logging.info("Model traced and saved as 'traced_model.pt'.")

        logging.info("Loading the traced model for CoreML conversion.")
        traced_model = torch.jit.load("traced_model.pt")
        logging.info("Traced model loaded successfully.")

        logging.info("Converting model to CoreML format with 16-bit quantization.")
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(shape=input_ids.shape, dtype=int, name="input_ids"),
                ct.TensorType(shape=attention_mask.shape, dtype=int, name="attention_mask"),
            ],
            outputs=[ct.TensorType(name="logits")],
            minimum_deployment_target=ct.target.iOS16
        )
        logging.info("CoreML model conversion completed.")

        logging.info("Saving CoreML model...")
        mlmodel.save(output_filename)
        logging.info(f"CoreML model saved to '{output_filename}'.")

        logging.info("Applying 8-bit quantization to reduce model size.")
        quantized_mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(mlmodel, nbits=8)
        quantized_filename = output_filename.replace(".mlmodel", "-quantized.mlmodel")
        quantized_mlmodel.save(quantized_filename)
        logging.info(f"Quantized CoreML model saved to '{quantized_filename}'.\n")

    except Exception as e:
        logging.error("Error occurred during the conversion process.")
        logging.error(f"Error: {e}")
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

    logging.info("=== Conversion Script Starting ===")
    try:
        convert_marianmt_to_coreml(args.model, tokenizer_name, args.example, args.output)
        logging.info("=== Conversion Script Completed Successfully ===")
    except Exception as e:
        logging.error("Fatal error during main execution.")
        logging.error(f"Error: {e}")
    logging.info("=== Script Finished ===\n")

if __name__ == "__main__":
    main()
