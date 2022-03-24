from models import LBDT_4

def get_model_by_name(model_name, *args, **kwargs):
    model = eval(model_name).JointModel(*args, **kwargs)
    return model
