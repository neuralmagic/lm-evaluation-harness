import copy
import json
import logging
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from packaging.version import Version

from lm_eval.logging.utils import _handle_non_serializable, remove_none_pattern


logger = logging.getLogger(__name__)


class ClearmlLogger:
    def __init__(self, **kwargs) -> None:
        """Attaches to clearml logger if already initialized. Otherwise, passes kwargs to clearml.Task.init()

        Args:
            kwargs Optional[Any]: Arguments for configuration.

        Parse and log the results returned from evaluator.simple_evaluate() with:
            wandb_logger.post_init(results)
            wandb_logger.log_eval_result()
            wandb_logger.log_eval_samples(results["samples"])
        """
        try:
            from clearml import Task

        except Exception as e:
            logger.warning(
                "To use the wandb reporting functionality please install wandb>=0.13.6.\n"
                "To install the latest version of wandb run `pip install wandb --upgrade`\n"
                f"{e}"
            )

        self.clearml_args: Dict[str, Any] = kwargs

        # initialize a task run
        self.clearml_task = Task.get_task(**self.clearml_args)
        if self.clearml_task is None:
            self.clearml_task = Task.init(**self.clearml_args)
        else:
            self.clearml_task.started()

    def post_init(self, results: Dict[str, Any]) -> None:
        self.results: Dict[str, Any] = copy.deepcopy(results)
        self.task_names: List[str] = list(results.get("results", {}).keys())
        self.group_names: List[str] = list(results.get("groups", {}).keys())

    def finish(self) -> None:
        self.clearml_task.mark_completed()

    def _sanitize_results_dict(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Sanitize the results dictionary."""
        _results = copy.deepcopy(self.results.get("results", dict()))

        # Remove None from the metric string name
        tmp_results = copy.deepcopy(_results)
        for task_name in self.task_names:
            task_result = tmp_results.get(task_name, dict())
            for metric_name, metric_value in task_result.items():
                _metric_name, removed = remove_none_pattern(metric_name)
                if removed:
                    _results[task_name][_metric_name] = metric_value
                    _results[task_name].pop(metric_name)
        return _results
    
    def _log_scalars(self) -> None:
        for task_name in self.task_names:
            task_result = self.clearml_results.get(task_name, dict())
            for metric_name, metric_value in task_result.items():
                self.clearml_task.get_logger().report_single_value(name=f"{task_name}/{metric_name}", value=metric_value)

    def _log_results_as_table(self) -> None:
        """Generate and log evaluation results as a table to W&B."""
        columns = [
            "Version",
            "Filter",
            "num_fewshot",
            "Metric",
            "Value",
            "Stderr",
        ]

        def make_table(columns: List[str], key: str = "results"):
            import wandb

            table = wandb.Table(columns=columns)
            results = copy.deepcopy(self.results)

            for k, dic in results.get(key).items():
                if k in self.group_names and not key == "groups":
                    continue
                version = results.get("versions").get(k)
                if version == "N/A":
                    version = None
                n = results.get("n-shot").get(k)

                for (mf), v in dic.items():
                    m, _, f = mf.partition(",")
                    if m.endswith("_stderr"):
                        continue
                    if m == "alias":
                        continue

                    if m + "_stderr" + "," + f in dic:
                        se = dic[m + "_stderr" + "," + f]
                        if se != "N/A":
                            se = "%.4f" % se
                        table.add_data(*[k, version, f, n, m, str(v), str(se)])
                    else:
                        table.add_data(*[k, version, f, n, m, str(v), ""])

            return table

        # log the complete eval result to W&B Table
        table = make_table(["Tasks"] + columns, "results")
        self.run.log({"evaluation/eval_results": table})

        if "groups" in self.results.keys():
            table = make_table(["Groups"] + columns, "groups")
            self.run.log({"evaluation/group_eval_results": table})

    def _log_results_as_artifact(self) -> None:
        self.clearml_task.upload_artifact(name='lm-evaluation-harness results', artifact_object=self.clearml_results)

    def log_eval_result(self) -> None:
        """Log evaluation results to ClearML."""
        # Log configs to ClearML
        self.clearml_results = self._sanitize_results_dict()
        # Log the evaluation metrics to ClearML
        self._log_scalars()
        # Log the evaluation metrics as ClearML Table
        #self._log_results_as_table()
        # Log the results dict as json to ClearML Artifacts
        self._log_results_as_artifact()

    '''
    def _generate_dataset(
        self, data: List[Dict[str, Any]], config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Generate a dataset from evaluation data.

        Args:
            data (List[Dict[str, Any]]): The data to generate a dataset for.
            config (Dict[str, Any]): The configuration of the task.

        Returns:
            pd.DataFrame: A dataframe that is ready to be uploaded to W&B.
        """
        ids = [x["doc_id"] for x in data]
        labels = [x["target"] for x in data]
        instance = [""] * len(ids)
        resps = [""] * len(ids)
        filtered_resps = [""] * len(ids)
        model_outputs = {}

        metrics_list = config["metric_list"]
        metrics = {}
        for metric in metrics_list:
            metric = metric.get("metric")
            if metric in ["word_perplexity", "byte_perplexity", "bits_per_byte"]:
                metrics[f"{metric}_loglikelihood"] = [x[metric][0] for x in data]
                if metric in ["byte_perplexity", "bits_per_byte"]:
                    metrics[f"{metric}_bytes"] = [x[metric][1] for x in data]
                else:
                    metrics[f"{metric}_words"] = [x[metric][1] for x in data]
            else:
                metrics[metric] = [x[metric] for x in data]

        if config["output_type"] == "loglikelihood":
            instance = [x["arguments"][0][0] for x in data]
            labels = [x["arguments"][0][1] for x in data]
            resps = [
                f'log probability of continuation is {x["resps"][0][0][0]} '
                + "\n\n"
                + "continuation will {} generated with greedy sampling".format(
                    "not be" if not x["resps"][0][0][1] else "be"
                )
                for x in data
            ]
            filtered_resps = [
                f'log probability of continuation is {x["filtered_resps"][0][0]} '
                + "\n\n"
                + "continuation will {} generated with greedy sampling".format(
                    "not be" if not x["filtered_resps"][0][1] else "be"
                )
                for x in data
            ]
        elif config["output_type"] == "multiple_choice":
            instance = [x["arguments"][0][0] for x in data]
            choices = [
                "\n".join([f"{idx}. {y[1]}" for idx, y in enumerate(x["arguments"])])
                for x in data
            ]
            resps = [np.argmax([n[0][0] for n in x["resps"]]) for x in data]
            filtered_resps = [
                np.argmax([n[0] for n in x["filtered_resps"]]) for x in data
            ]
        elif config["output_type"] == "loglikelihood_rolling":
            instance = [x["arguments"][0][0] for x in data]
            resps = [x["resps"][0][0] for x in data]
            filtered_resps = [x["filtered_resps"][0] for x in data]
        elif config["output_type"] == "generate_until":
            instance = [x["arguments"][0][0] for x in data]
            resps = [x["resps"][0][0] for x in data]
            filtered_resps = [x["filtered_resps"][0] for x in data]

        model_outputs["raw_predictions"] = resps
        model_outputs["filtered_predictions"] = filtered_resps

        df_data = {
            "id": ids,
            "data": instance,
        }
        if config["output_type"] == "multiple_choice":
            df_data["choices"] = choices

        tmp_data = {
            "input_len": [len(x) for x in instance],
            "labels": labels,
            "output_type": config["output_type"],
        }
        df_data.update(tmp_data)
        df_data.update(model_outputs)
        df_data.update(metrics)

        return pd.DataFrame(df_data)

    def _log_samples_as_artifact(
        self, data: List[Dict[str, Any]], task_name: str
    ) -> None:
        import wandb

        # log the samples as an artifact
        dumped = json.dumps(
            data,
            indent=2,
            default=_handle_non_serializable,
            ensure_ascii=False,
        )
        artifact = wandb.Artifact(f"{task_name}", type="samples_by_task")
        with artifact.new_file(
            f"{task_name}_eval_samples.json", mode="w", encoding="utf-8"
        ) as f:
            f.write(dumped)
        self.run.log_artifact(artifact)
        # artifact.wait()

    def log_eval_samples(self, samples: Dict[str, List[Dict[str, Any]]]) -> None:
        """Log evaluation samples to W&B.

        Args:
            samples (Dict[str, List[Dict[str, Any]]]): Evaluation samples for each task.
        """
        task_names: List[str] = [
            x for x in self.task_names if x not in self.group_names
        ]

        ungrouped_tasks = []
        tasks_by_groups = {}

        for task_name in task_names:
            group_names = self.task_configs[task_name].get("group", None)
            if group_names:
                if isinstance(group_names, str):
                    group_names = [group_names]

                for group_name in group_names:
                    if not tasks_by_groups.get(group_name):
                        tasks_by_groups[group_name] = [task_name]
                    else:
                        tasks_by_groups[group_name].append(task_name)
            else:
                ungrouped_tasks.append(task_name)

        for task_name in ungrouped_tasks:
            eval_preds = samples[task_name]

            # log the samples as a W&B Table
            df = self._generate_dataset(eval_preds, self.task_configs.get(task_name))
            self.run.log({f"{task_name}_eval_results": df})

            # log the samples as a json file as W&B Artifact
            self._log_samples_as_artifact(eval_preds, task_name)

        for group, grouped_tasks in tasks_by_groups.items():
            grouped_df = pd.DataFrame()
            for task_name in grouped_tasks:
                eval_preds = samples[task_name]
                df = self._generate_dataset(
                    eval_preds, self.task_configs.get(task_name)
                )
                df["group"] = group
                df["task"] = task_name
                grouped_df = pd.concat([grouped_df, df], ignore_index=True)

                # log the samples as a json file as W&B Artifact
                self._log_samples_as_artifact(eval_preds, task_name)

            self.run.log({f"{group}_eval_results": grouped_df})
    '''