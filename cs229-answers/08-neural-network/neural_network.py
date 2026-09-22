import json
from pathlib import Path


def example_weights():
    """Initialize the network weight dictionary."""
    w = {}

    w['hidden_layer_0_1'] = 0
    w['hidden_layer_1_1'] = 0
    w['hidden_layer_2_1'] = 0
    w['hidden_layer_0_2'] = 0
    w['hidden_layer_1_2'] = 0
    w['hidden_layer_2_2'] = 0
    w['hidden_layer_0_3'] = 0
    w['hidden_layer_1_3'] = 0
    w['hidden_layer_2_3'] = 0

    w['output_layer_0'] = 0
    w['output_layer_1'] = 0
    w['output_layer_2'] = 0
    w['output_layer_3'] = 0

    return w


def optimal_step_weights():
    """Weights for the step-activation network in PS3, problem 1."""
    w = example_weights()

    w['hidden_layer_0_1'] = -4.0
    w['hidden_layer_1_1'] = 1.0
    w['hidden_layer_2_1'] = 1.0
    w['hidden_layer_0_2'] = -0.5
    w['hidden_layer_1_2'] = 0
    w['hidden_layer_2_2'] = 1.0
    w['hidden_layer_0_3'] = -0.5
    w['hidden_layer_1_3'] = 1.0
    w['hidden_layer_2_3'] = 0

    w['output_layer_0'] = -3
    w['output_layer_1'] = 1
    w['output_layer_2'] = 1
    w['output_layer_3'] = 1

    return w


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    weights = optimal_step_weights()
    (output / "step_weights.json").write_text(json.dumps(weights, indent=2) + "\n")
    print(json.dumps(weights, indent=2))
