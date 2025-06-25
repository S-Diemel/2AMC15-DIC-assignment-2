import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# csv files
dqn_csv = "metrics/dqn__metrics_new.csv"
ppo_csv = "metrics/ppo__metrics.csv"

# agent color same as in other plots
agent_info = {
    "DQN": {"file": dqn_csv, "color": "orange"},
    "PPO": {"file": ppo_csv, "color": "green"},
}

# load files
dfs = []
for agent, info in agent_info.items():
    df = pd.read_csv(info["file"])
    df["agent"] = agent
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

# sort so that phase-stage ordering is consistent
df = df.sort_values(["phase_number", "eval_stage"])

# map numeric stages to names
stage_map = {0: "start", 1: "middle", 2: "end"}
df["stage_name"] = df["eval_stage"].map(stage_map)

# build x-axis labels
df["phase_stage"] = (
    df["phase_number"].astype(int).astype(str)
    + "-"
    + df["stage_name"]
)
x_labels = df["phase_stage"].unique()
x_pos = np.arange(len(x_labels))

# plot
plt.figure(figsize=(15, 4))
for agent, info in agent_info.items():
    sub = df[df["agent"] == agent]
    plt.plot(
        x_pos,
        sub["success_rate"].values,
        marker="o",
        color=info["color"],
        label=f"{agent}: Success Rate",
    )
    plt.plot(
        x_pos,
        sub["avg_percent_delivered"].values,
        marker="^",
        linestyle="--",    
        color=info["color"],
        alpha=0.4,
        label=f"{agent}: Items Delivered Rate",
    )

plt.title("Success Rate and Items Delivered Rate n vs. Phase-Stage")
plt.xlabel("Phase-Stage")
plt.ylabel("Success Rate")
plt.ylim(0, 1.05)
plt.grid(True)
plt.xticks(x_pos, x_labels, rotation=45)
plt.legend()
plt.tight_layout()
# save to file
plt.savefig("metrics/training_metrics_plot.png", dpi=600, bbox_inches="tight")
