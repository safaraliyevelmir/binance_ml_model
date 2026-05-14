import re
import pandas as pd

def parse_optuna_logs(file_paths):
    all_data = []
    
    # Parametrləri tutmaq üçün Regex
    trial_start_re = re.compile(r"Trial\s+(\d+)\s+\|\s+bar=([\w\d]+)\s+pt/sl=([\d._]+)span=(\d+)\s+hold=(\d+)\s+cusum=(\w+)\s+max_depth=(\d+)")
    # Fold nəticələrini tutmaq üçün Regex
    fold_re = re.compile(r"fold (\d+):train=[\d,]+\s+test=[\d,]+train_acc=([\d.]+)\s+acc=([\d.]+)f1_macro=([\d.]+)f1_long=([\d.]+)f1_flat=([\d.]+)prec_long=([\d.]+)rec_long=([\d.]+)sharpe=([-\d.]+)")

    for file_path in file_paths:
        try:
            with open(f"logs/{file_path}", 'r') as f:
                current_params = None
                for line in f:
                    trial_match = trial_start_re.search(line)
                    if trial_match:
                        current_params = {
                            "trial_id": trial_match.group(1),
                            "bar_threshold": trial_match.group(2),
                            "tp_sl": trial_match.group(3),
                            "span": trial_match.group(4),
                            "max_bar": trial_match.group(5),
                            "max_depth": trial_match.group(7),
                        }
                        continue
                    
                    fold_match = fold_re.search(line)
                    if fold_match and current_params:
                        row = current_params.copy()
                        row.update({
                            "fold_no": fold_match.group(1),
                            "train_acc": fold_match.group(2),
                            "test_acc": fold_match.group(3),
                            "f1_macro": fold_match.group(4),
                            "f1_long": fold_match.group(5),
                            "sharpe": fold_match.group(9)
                        })
                        print(fold_match)
                        all_data.append(row)
        except FileNotFoundError:
            print(f"Fayl tapılmadı: {file_path}")
                    
    return pd.DataFrame(all_data)

# Faylların siyahısı
files = [
    'optuna_dollar_bar_gridsearch_20260506_124715.log'
]

df = parse_optuna_logs(files)
df.to_csv('optuna_results_full.csv', index=False)
print("Uğurla tamamlandı! 'optuna_results_full.csv' faylı yaradıldı.")