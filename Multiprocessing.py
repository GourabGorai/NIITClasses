import pandas as pd
from deep_translator import GoogleTranslator
import multiprocessing as mp

def translate_text(text, target_lang, service='google', api_key=None):
    try:
        if service == 'google':
            # Automatically detect source language
            translator = GoogleTranslator(target=target_lang)
            translated_text = translator.translate(text)
        else:
            raise ValueError("Unsupported translation service or missing API key for Microsoft Translator")

        return translated_text

    except Exception as e:
        return f"An error occurred: {str(e)}"

def process_chunk(df_chunk):
    # Example processing function
    df_chunk['translated'] = df_chunk['spec'].apply(lambda x: translate_text(x, 'en'))
    return df_chunk

def worker(process_idx, df_chunk, return_dict):
    processed_chunk = process_chunk(df_chunk)
    return_dict[process_idx] = processed_chunk

def main():
    file_path = 'orders (1).csv'
    df = pd.read_csv(file_path)

    num_processes = mp.cpu_count()
    chunk_size = 200  # Number of rows each process will handle

    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for i in range(0, len(df), chunk_size):
        df_chunk = df.iloc[i:i+chunk_size]

        p = mp.Process(target=worker, args=(i, df_chunk, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    processed_df = pd.concat(return_dict.values())
    processed_df.to_csv('D:\\_igetintopc.com_App_Builder_2018processed_file.csv', index=False)

if __name__ == "__main__":
    main()
