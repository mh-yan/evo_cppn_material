import neat
import os

exec_times = 3
base_pth = "./all_data3"
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "config.ini")


tasks_parameters = [
    # {
    #     "name": "sym2_task",
    #     "pcdtype": "sym2",
    #     "times": exec_times,
    #     "out_path": f"{base_pth}/sym2",
    #     "tradeoff": [
    #         "Max:E-Min:nu",
    #         "Max:E-Max:nu",
    #     ],
    # },
    # {
    #     "name": "sym2_tensor_task",
    #     "pcdtype": "sym2",
    #     "times": 1,
    #     "out_path": f"{base_pth}/sym2",
    #     "tradeoff": [
    #         "tensor1up",
    #         "tensor1down",
    #         "tensor2up",
    #         "tensor2down",
    #         "tensor3up",
    #         "tensor3down",
    #     ],
    # },
    {
        "name": "sym4_task",
        "pcdtype": "sym4",
        "times": exec_times,
        "out_path": f"{base_pth}/sym4",
        "tradeoff": ["Max:E-Min:nu", "Max:E-Max:nu"],
    },
    {
        "name": "rotate_task",
        "pcdtype": "rotate",
        "times": exec_times,
        "out_path": f"{base_pth}/rotate",
        "tradeoff": ["Max:E-Min:nu", "Max:E-Max:nu"],
    },
    {
        "name": "parallel_task",
        "pcdtype": "parallel",
        "times": exec_times,
        "out_path": f"{base_pth}/parallel",
        "tradeoff": ["Max:E-Min:nu", "Max:E-Max:nu"],
    },
    # {
    #     "name": "nosym_task",
    #     "pcdtype": "nosym",
    #     "times": exec_times,
    #     "out_path": f"{base_pth}/nosym",
    #     "tradeoff": ["shear_normal", "shear_bulk"],
    # },
]


# tasks_parameters = [
#     {
#         "name": "sym2_tensor_task",
#         "pcdtype": "sym2",
#         "times": 1,
#         "out_path": f"{base_pth}/sym2",
#         "tradeoff": [
#             "tensor1up",
#             "tensor2leftdown",
#             "tensor3leftup",
#             "tensor3leftdown",
#         ],
#     },
# ]


class exc_all(object):
    def __init__(self, para_dict, experiment) -> None:
        self.para_dict = para_dict
        self.tasks = []
        self.experiment = experiment
        self.define_tasks()

    def define_tasks(self):
        for params in self.para_dict:
            #             initiate et
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_path,
            )
            config.pcdtype = params["pcdtype"]
            et = executor_tasks(
                config=config,
                name=params["name"],
                times=params["times"],
                out_path=params["out_path"],
                tradeoff=params["tradeoff"],
            )
            #             append into tasks
            self.tasks.append(et)

    def re_init(self):
        pass

    def run_all(self):
        #        每一个案例的每一个tradeoff 算 n次
        for et in self.tasks:
            for tf in et.tradeoff:
                for t in range(et.times):
                    et.run(t, tf, self.experiment)
        print("======================Finish all======================")


class executor_tasks(object):
    def __init__(self, config, name, times, out_path, tradeoff) -> None:
        self.config = config
        self.name = name
        self.times = times
        self.out_path = out_path
        self.tradeoff = tradeoff

    def run(self, t, cur_tradeoff, experiment):
        print(
            "Current Task Details:\n"
            f"  Name    : {self.name}\n"
            f"  CurTimes   : {t}\n"
            f"  PointCloud Type: {self.config.pcdtype}\n"
            f"  Current Trade off: {cur_tradeoff}\n"
        )
        # 执行程序
        experiment(
            config=self.config,
            task_name=self.name + "_" + cur_tradeoff,
            cur_tradeoff=cur_tradeoff,
            out_path=self.out_path,
            cur_times=t,
            n_generations=100,
        )

    def __str__(self) -> str:
        return (
            "Task Details:\n"
            f"  Name    : {self.name}\n"
            f"  Times   : {self.times}\n"
            f"  PointCloud Type: {self.config.pcdtype}\n"
            f"  Trade off: {self.tradeoff}\n"
        )
