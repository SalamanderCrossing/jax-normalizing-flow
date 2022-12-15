import mate
from ..trainers.flow_trainer import TrainerModule
from ..models.multiscale_flow import create_multiscale_flow


multiscale_flow = create_multiscale_flow()
if mate.is_train:
    trainer = TrainerModule(multiscale_flow)


    
