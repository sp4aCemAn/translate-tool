import os
import sys
import ctranslate2
import transformers

# --- Configuration ---
# Uses the same converted model file as before
model_path = os.environ.get("CT2_CONVERTED_MODEL", "/app/models/nllb-200-3.3B-ct2-int8") # Or nllb-200-3.3B-ct2-int8 if you used that
hf_model_name = os.environ.get("HF_MODEL_NAME", "facebook/nllb-200-3.3B") # Or facebook/nllb-200-3.3B

device = "cuda" # Use "cuda" or "cpu"

# NLLB Language Codes
eng_code = "eng_Latn"
jpn_code = "jpn_Jpan"

# --- Load Models ---
translator = None
tokenizer_en = None # Tokenizer configured for English source
tokenizer_ja = None # Tokenizer configured for Japanese source

try:
    print(f"Loading CTranslate2 model from: {model_path} on device: {device}")
    translator = ctranslate2.Translator(model_path, device=device)

    print(f"Loading HuggingFace tokenizers: {hf_model_name}")
    # Load tokenizer configured for English source
    tokenizer_en = transformers.AutoTokenizer.from_pretrained(
        hf_model_name,
        src_lang=eng_code,
        legacy=False
    )
    # Load tokenizer configured for Japanese source
    tokenizer_ja = transformers.AutoTokenizer.from_pretrained(
        hf_model_name,
        src_lang=jpn_code,
        legacy=False
    )

    print("\nModels loaded successfully. Ready for translation.")
    print("Type text to translate.")
    print("Type /switch to change direction.")
    print("Type '/quit' or '/exit' to finish.")
    print("-" * 30)

except Exception as e:
    print(f"FATAL: Error loading models: {e}", file=sys.stderr)
    sys.exit(1)

# --- Translation State ---
current_direction = "en->ja" # Start with English to Japanese

# --- Translation Loop ---
while True:
    try:
        # Set codes and tokenizer based on current direction
        if current_direction == "en->ja":
            src_lang_code = eng_code
            tgt_lang_code = jpn_code
            current_tokenizer = tokenizer_en
            prompt_lang = "English"
            target_lang_name = "Japanese"
        else: # ja->en
            src_lang_code = jpn_code
            tgt_lang_code = eng_code
            current_tokenizer = tokenizer_ja
            prompt_lang = "Japanese"
            target_lang_name = "English"

        # Get input from the user
        print(f"\nEnter text to translate ({prompt_lang} -> {target_lang_name}) (or type /switch):")
        input_text = input("> ")
        input_lower_stripped = input_text.lower().strip()

        # --- Command Handling ---
        if input_text == '/switch':
            current_direction = "ja->en" if current_direction == "en->ja" else "en->ja"
            print(f"Direction switched. Now translating {current_direction.replace('->', ' to ')}.")
            print("-" * 30)
            continue # Go back to prompt for new input

        if input_text == '/exit':
            print("Exiting translator.")
            break

        # Handle empty input
        if not input_text.strip():
            continue

        # --- Prepare Source Tokens ---
        # Use the tokenizer configured for the current source language
        inputs = current_tokenizer(input_text, return_tensors=None, add_special_tokens=True)
        source_tokens = current_tokenizer.convert_ids_to_tokens(inputs["input_ids"])
        print(f"DEBUG: Source Tokens ({src_lang_code}): {source_tokens}")

        if not source_tokens:
             print("Warning: Input resulted in empty tokens.")
             continue

        # --- Prepare Target Prefix Tokens ---
        # Can use either tokenizer here as vocab is shared, but use current for consistency
        raw_prefix_token_ids = current_tokenizer.encode(tgt_lang_code, add_special_tokens=False)
        target_prefix_token_strings = current_tokenizer.convert_ids_to_tokens(raw_prefix_token_ids)

        # --- DEBUGGING OUTPUT for Prefix ---
        print(f"DEBUG: Target Lang Code: '{tgt_lang_code}'")
        print(f"DEBUG: Raw Prefix Token IDs: {raw_prefix_token_ids}")
        print(f"DEBUG: Prefix Token Strings: {target_prefix_token_strings}")
        # --- End DEBUGGING ---

        if not target_prefix_token_strings:
            print(f"Error: Could not generate target prefix tokens for language code '{tgt_lang_code}'. Skipping translation.", file=sys.stderr)
            continue

        # --- Translate using CTranslate2 ---
        results = translator.translate_batch(
            [source_tokens],
            beam_size=4,
            max_decoding_length=256,
            target_prefix=[target_prefix_token_strings] # Force decoder to start with target lang code
       )

        # --- Process Output ---
        # Get the list of output token STRINGS from the best hypothesis
        output_token_strings = results[0].hypotheses[0]
        print(f"DEBUG: Output Token Strings: {output_token_strings}")

        # Convert the token STRINGS back to integer IDs
        # Use the current_tokenizer, NLLB vocab is shared so it works for decoding either lang
        output_token_ids_integers = current_tokenizer.convert_tokens_to_ids(output_token_strings)
        print(f"DEBUG: Output Token IDs (Integers): {output_token_ids_integers}")

        # Decode the integer IDs, skipping special tokens
        target_text = current_tokenizer.decode(output_token_ids_integers, skip_special_tokens=True)

        # Print the final translation
        print(f"Translation ({target_lang_name}): {target_text}")
        print("-" * 30)

    except EOFError:
        print("\nExiting translator (EOF detected).")
        break
    except KeyboardInterrupt:
        print("\nExiting translator (Interrupt detected).")
        break
    except Exception as e:
        print(f"\nAn error occurred during translation: {e}", file=sys.stderr)
        print("-" * 30) 
