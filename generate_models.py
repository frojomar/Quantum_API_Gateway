from estimation_model import generate_all_models



if __name__ == "__main__":
    print("[INIT] Generating estimation of waiting time's models...")
    result = generate_all_models()
    if result:
        print("[END] Models generated.")
    else:
        print("[END] Models not generated.")
