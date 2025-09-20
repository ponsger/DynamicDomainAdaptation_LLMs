#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

# === CONFIGURATION ===
FILENAME = "context.txt"
SIMILARITY_THRESHOLD = 0.5 # Steering strength

# === LOGGING FUNCTION ===
def write_log(message, log_file="log.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}\n"
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(full_message)

# === LOAD CONTEXT ===
def read_file_lines(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return [line.strip().lower() for line in file.readlines()]
    
def load_questions(filepath):
    questions = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                questions.append(line)
    return questions

def run_batch(filepath, alpha_values=[0.1, 0.5, 0.9]):
    questions = load_questions(filepath)
    for prompt in questions:
        for alpha in alpha_values:
            write_log("-------------------------------------------------------------------------------------------")
            write_log(f"\nRunning: '{prompt}' with ALPHA={alpha}")
            print(f"Running: '{prompt}' with ALPHA={alpha}")
            main(prompt, ALPHA=alpha)


def main(prompt, ALPHA=1):
    contexts = read_file_lines(FILENAME)
    prompt = prompt.lower()  # Ensure prompt is lowercase
    write_log(f"Similarity threshold set to {SIMILARITY_THRESHOLD}")
    write_log(f"Steering strength (ALPHA) set to {ALPHA}")

    # === EMBEDDING MODEL ===
    embedding_model = SentenceTransformer("all-mpnet-base-v2")  # 768-dim output
    context_embeddings = embedding_model.encode(contexts, convert_to_tensor=True)
    prompt_embedding = embedding_model.encode(prompt, convert_to_tensor=True)

    # === FIND RELEVANT CONTEXT ===
    similarities = []
    for ctx, ctx_emb in zip(contexts, context_embeddings):
        score = util.cos_sim(prompt_embedding, ctx_emb).item()
        similarities.append((score, ctx_emb))

    relevant_embeddings = [emb for score, emb in similarities if score >= SIMILARITY_THRESHOLD]
    if not relevant_embeddings:
        write_log("No relevant context found.")
        return

    # === COMPUTE STEERING VECTOR ===
    steering_vector = torch.stack(relevant_embeddings).mean(dim=0)
    similarity = util.cos_sim(prompt_embedding, steering_vector).item()
    if similarity < SIMILARITY_THRESHOLD:
        write_log("Context is NOT valid — skipping generation.")
        return

    write_log(f"Context is valid. Similarity score: {similarity}")

    # === DEVICE SETUP (Apple Silicon) ===
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # === LOAD PHI-2 ===
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
    model.to(device)
    model.eval()

    # === TOKENIZE PROMPT ===
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]  # shape: [1, seq_len]

    # === GET EMBEDDINGS ===
    embedding_layer = model.get_input_embeddings()  # Embedding dim: 2048
    input_embeddings = embedding_layer(input_ids)   # shape: [1, seq_len, 2048]

    # === PROJECT STEERING VECTOR (768 → what is needed) ===
    target_embedding_dim = model.get_input_embeddings().embedding_dim
    write_log(f"Detected embedding dim: {target_embedding_dim}")
    projection_layer = torch.nn.Linear(768, target_embedding_dim).to(device)

    # projection_layer = torch.nn.Linear(768, 2048).to(device)
    steering_vector_projected = projection_layer(steering_vector.to(device))

    write_log(f"Input embeddings shape: {input_embeddings.shape}")
    write_log(f"Steering vector projected shape: {steering_vector_projected.shape}")

    # === APPLY STEERING TO FIRST TOKEN ===
    input_embeddings[0, 0] = (1 - ALPHA) * input_embeddings[0, 0] + ALPHA * steering_vector_projected

    # === Create attention mask manually ===
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(device)

    # === Generate output ===
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            max_new_tokens=100,
            num_beams=4,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )


    # # === GENERATE OUTPUT ===
    # with torch.no_grad():
    #     outputs = model.generate(
    #         inputs_embeds=input_embeddings,
    #         max_new_tokens=100,
    #         num_beams=4,
    #         early_stopping=True
    #     )

    # === DECODE AND PRINT RESULT ===
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    write_log("Generated response: \n" + response)

if __name__ == "__main__":
    try:
        run_batch('questions.txt')
    except Exception as e:
        write_log(f"Error{str(e)}")
    finally:
        write_log("\n------------------------------------------------------------------------------------------\n")

