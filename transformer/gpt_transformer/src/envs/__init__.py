
import pyrootutils

path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
pyrootutils.set_root(path = path,
                     project_root_env_var = True,
                     dotenv = True,
                     pythonpath = True)

from transformer.gpt_transformer.src.envs.registration import register_environments

registered_environments = register_environments()

