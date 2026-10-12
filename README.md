# Fair Fairness Benchmark: course project workspace

This repository organizes the published [Fair Fairness Benchmark (FFB)](https://github.com/ahxt/fair_fairness_benchmark) code alongside our group's reproduction work and supporting tools. The original benchmark is a framework for evaluating in-processing group fairness methods. Our work here is an extension and organization of that code, not a claim that we reproduced every result in the paper.

## Progress at a glance

| Workstream | What is in the repository | Current status |
| --- | --- | --- |
| FFB reference code | Original training scripts, models, metrics, dataset instructions, tutorial, and figures under [`ffb/`](ffb/) | Available as the baseline; the original paper's figures and running logs are upstream results, not new group results. |
| CelebA reproduction | Four ResNet-18/ResNet-20 runners for ERM and DiffDP under [`ffb/reproduce/`](ffb/reproduce/) | Scripts and a dependency/data preflight are implemented. No completed run output is committed; training still needs the CelebA data and dependencies. |
| PDF evaluation-metric extractor | [`team/eval_metrics_tool/`](team/eval_metrics_tool/) | Rule-based PDF text/table extraction, metric matching, and ranking are implemented. Scanned PDFs and some table layouts remain unsupported. |
| Metric record initializer | [`team/initialize_metrics_record_tool/`](team/initialize_metrics_record_tool/) | Extracts reported values into a metric-to-value record. Ambiguous or missing values remain empty for review. |
| Local repository agent | [`team/agent/`](team/agent/) | A basic Ollama/LangChain agent can list the available reproduction scripts. It is not yet an automated experiment runner. |

## Repository map

```text
.
├── README.md                     Project status and entry points
├── LICENSE                       MIT license inherited from FFB
├── ffb/                          Published FFB code and our reproduction work
│   ├── src/                      Original model, training, and metric code
│   ├── benchmark/                Original benchmark helper
│   ├── datasets/                 Dataset instructions (data not committed)
│   ├── tutorial/                 Original notebook
│   ├── img/                      Original paper/README figures
│   ├── reproduce/                Our CelebA reproduction runners
│   ├── requirements.txt          Original FFB dependency list
│   └── readme.md                 Original FFB documentation and historical logs
└── team/                         Group-built supporting tools
    ├── agent/                    Local repository agent
    ├── eval_metrics_tool/        PDF evaluation-metric extractor
    └── initialize_metrics_record_tool/  Paper-reported metric values
```

The code in `ffb/src/`, `ffb/benchmark/`, `ffb/datasets/`, `ffb/tutorial/`, and `ffb/img/` comes from the [upstream FFB project](https://github.com/ahxt/fair_fairness_benchmark). The `ffb/reproduce/` scripts and folders under `team/` are this group's additions. See the [FFB paper](https://arxiv.org/abs/2306.09468) for the benchmark's methods and published results.

## Development milestones

| Date | Repository evidence | Contribution |
| --- | --- | --- |
| Sep 20–21, 2026 | [CelebA runners](https://github.com/Auspr7ng/fair_fairness_benchmark/commit/b1c24bbae4e37feafde39fdf1d2eecc565d4a9f7) and [directory organization](https://github.com/Auspr7ng/fair_fairness_benchmark/commit/7b2e45121c2cc10fc146e57f140fd82b22156e77) | Yuchao added the partial ResNet reproduction scripts. |
| Sep 21, 2026 | [Local agent](https://github.com/Auspr7ng/fair_fairness_benchmark/commit/8ae3588d3b0790a388022c9aa760f9c3b6173d23) and [instructions](https://github.com/Auspr7ng/fair_fairness_benchmark/commit/b5877e2b480ea0e3abb132c60b95afdc20d18216) | Aarush added the Ollama/LangChain demonstration agent. |
| Oct 4, 2026 | [Metric extractor](https://github.com/Auspr7ng/fair_fairness_benchmark/commit/044351ccfdc57b8c7ed4ea2d14d1b76baa7aef00) and [fairness metric patterns](https://github.com/Auspr7ng/fair_fairness_benchmark/commit/c7b8cc8582231bc1c487d540b731f84a4c42f6ca) | Andrew added PDF extraction and expanded the metric dictionary. |

## Try the current work

Run the following commands from the repository root. Each component has its own requirements; the data and Ollama model are not included in the repository.

### Check CelebA reproduction readiness

```bash
python ffb/reproduce/reproduce_resnet18_celeba.py --check
```

This prints whether `ffb/datasets/celeba/raw/celeba.csv` and the required Python packages are present. To prepare the data, follow [`ffb/datasets/readme.md`](ffb/datasets/readme.md). After preparation, the ERM and DiffDP entry points include:

```bash
python ffb/reproduce/reproduce_resnet18_celeba.py --weights random
python ffb/reproduce/reproduce_diffdp_resnet18_celeba.py --weights random
python ffb/reproduce/reproduce_resnet20_celeba.py
python ffb/reproduce/reproduce_diffdp_resnet20_celeba.py
```

These are partial reproduction runners with a default 20,000-row sample and 150 training steps. They write local outputs under `ffb/results/`, which is ignored by Git. We have not committed numerical results, so please do not interpret the published FFB results in [`ffb/readme.md`](ffb/readme.md) as results from these runs. The original `ffb/requirements.txt` pins CUDA-specific PyTorch builds; those pins may need platform-specific installation on macOS or a non-CUDA machine.

### Extract evaluation metrics from a paper

```bash
python -m pip install -r team/eval_metrics_tool/requirements.txt
cd team/eval_metrics_tool
python -m eval_metrics /path/to/paper.pdf --top-k 5
```

The extractor returns ranked metric names based on mentions in the PDF; it does not compute benchmark scores. Its [README](team/eval_metrics_tool/README.md) explains the scoring rules and limitations. The development requirements list test dependencies, but no test suite is currently committed.

### Initialize a paper metric record

```bash
python -m team.initialize_metrics_record_tool /path/to/paper.pdf --top-k 5
python -m team.initialize_metrics_record_tool /path/to/FFB.pdf --metric Accuracy --metric F1 --row-label "Bank Age"
```

This returns JSON with one reported value per metric where the paper gives a clear value. Use `--metric` to choose names or `--row-label` to select a model row in a table. See the [tool README](team/initialize_metrics_record_tool/README.md) for the Python API and limitations.

### Run the local agent

The agent needs Python 3.12, a running [Ollama](https://ollama.com/) service, and the `qwen3:4b` model. From the repository root:

```bash
ollama pull qwen3:4b
python -m pip install -r team/agent/requirements.txt
python team/agent/basic_agent.py
```

The agent currently answers one demonstration question by calling a tool that lists files in `ffb/reproduce/`. See its [README](team/agent/README.md) for the recommended virtual environment setup.

## Attribution

FFB was introduced by Xiaotian Han and colleagues in [*FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods*](https://arxiv.org/abs/2306.09468). This repository is a fork of their [public implementation](https://github.com/ahxt/fair_fairness_benchmark) and retains its MIT [license](LICENSE). The historical experiment counts and plots in the original documentation belong to the FFB authors.
