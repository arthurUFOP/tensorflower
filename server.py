import flwr as fl
import tensorflow as tf
import time
import os
import csv
from datetime import datetime

HOST                  = "127.0.0.1"
PORT                  = "7517"
GLOBAL_EPOCHS         = 6
VERBOSE               = 1
OUTPUT_DIR            = "ColorretalExperimentLogs"
FRACTION_FIT          = 1
FRACTION_EVALUATE     = 1
MIN_FIT_CLIENTS       = 4
MIN_EVALUATE_CLIENTS  = 4
MIN_AVAILABLE_CLIENTS = 4
DECAY_ROUND_1         = 1
DECAY_ROUND_2         = 5
DECAY_FACTOR          = 0.9

if os.path.exists(os.path.join(os.curdir, "LOGS", OUTPUT_DIR)):
  print("ERROR: Output Dir Already Exists!")
  exit(1)
os.mkdir(os.path.join(os.curdir, "LOGS", OUTPUT_DIR))

def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    aucs = [num_examples * m["auc"] for num_examples, m in metrics]
    precs = [num_examples * m["precision"] for num_examples, m in metrics]
    recs = [num_examples * m["recall"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples),
             "auc": sum(aucs) / sum(examples),
             "precision": sum(precs) / sum(examples),
             "recall": sum(recs) / sum(examples)}

def fit_config(server_round):
    decay = True if (server_round==DECAY_ROUND_1 or server_round==DECAY_ROUND_2) else False

    config = {
        "lr_decay" : str(decay),
        "decay_factor": str(DECAY_FACTOR),
        "alter_trainable": str(False),
        "trainable" : str(True),
    }

    return config

strategy = fl.server.strategy.FedAvg(
    fraction_fit=FRACTION_FIT,
    fraction_evaluate=FRACTION_EVALUATE,
    min_fit_clients=MIN_FIT_CLIENTS,
    min_evaluate_clients=MIN_EVALUATE_CLIENTS,
    min_available_clients=MIN_AVAILABLE_CLIENTS,
    evaluate_metrics_aggregation_fn=weighted_average,
    on_fit_config_fn=fit_config)

print("Starting server...")
start = time.time()
history = fl.server.start_server(server_address=f"{HOST}:{PORT}",
                                config=fl.server.ServerConfig(num_rounds=GLOBAL_EPOCHS),
                                strategy=strategy)
end = time.time()
print("Training Ended! Writing Logs...")

aucs = []
accs = []
precs = []
recs = []
for _, auc_val in history.metrics_distributed['auc']:
    aucs.append(auc_val)
for _, acc_val in history.metrics_distributed['accuracy']:
    accs.append(acc_val)
for _, pre_val in history.metrics_distributed['precision']:
    precs.append(pre_val)
for _, rec_val in history.metrics_distributed['recall']:
    recs.append(rec_val)

with open(os.path.join(os.curdir, "LOGS", OUTPUT_DIR, "config.log"), "w+") as f:
    f.write(f"Today's date: {datetime.now()}\n")
    f.write("The training was configured as follows:\n")
    f.write(f"""
            HOST                  = {HOST                 }
            PORT                  = {PORT                 }
            GLOBAL_EPOCHS         = {GLOBAL_EPOCHS        }
            VERBOSE               = {VERBOSE              }
            OUTPUT_DIR            = {OUTPUT_DIR           }
            FRACTION_FIT          = {FRACTION_FIT         }
            FRACTION_EVALUATE     = {FRACTION_EVALUATE    }
            MIN_FIT_CLIENTS       = {MIN_FIT_CLIENTS      }
            MIN_EVALUATE_CLIENTS  = {MIN_EVALUATE_CLIENTS }
            MIN_AVAILABLE_CLIENTS = {MIN_AVAILABLE_CLIENTS}
            DECAY_ROUND_1         = {DECAY_ROUND_1        }
            DECAY_ROUND_2         = {DECAY_ROUND_2        }
            DECAY_FACTOR          = {DECAY_FACTOR         }
            """)

with open(os.path.join(os.curdir, "LOGS", OUTPUT_DIR, "execution.log"), "w+") as f:
    f.write(str(history))

with open(os.path.join(os.curdir, "LOGS", OUTPUT_DIR, "report_top3_and_best.csv"), "a+", newline='') as f:
    writer = csv.writer(f)

    sorted_aucs = sorted(aucs, reverse=True)
    sorted_accs = sorted(accs, reverse=True)
    sorted_precs = sorted(precs, reverse=True)
    sorted_recs = sorted(recs, reverse=True)

    data = [end - start,
            sum(sorted_aucs[:3])/3, sum(sorted_accs[:3])/3,
            sum(sorted_precs[:3])/3, sum(sorted_recs[:3])/3,
            sorted_aucs[0], sorted_accs[0],
            sorted_precs[0], sorted_recs[0]]

    writer.writerow(['time', 'top3-auc', 'top3-acc', 'top3-prec', 'top3-recs', 'best-auc', 'best-acc', 'best-prec', 'best-rec'])
    writer.writerow(data)

with open(os.path.join(os.curdir, "LOGS", OUTPUT_DIR, "report_each_epoch.csv"), "a+", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "auc", "acc", "prec", "rec"])

    for i in range(len(aucs)):
      writer.writerow([i+1, aucs[i], accs[i], precs[i], recs[i]])

print("Logs Written! All done, ending now...")