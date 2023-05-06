import datetime as dt

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.dataset.transformer import *
from utils import get_mnist
import matplotlib.pyplot as plt
from pyspark import SparkContext
from matplotlib.pyplot import imshow

sc=SparkContext.getOrCreate(conf=create_spark_conf().setMaster("local[4]").set("spark.driver.memory","2g"))
init_engine()

mnist_path = "datasets/mnist"
(train_data, test_data) = get_mnist(sc, mnist_path)
learning_rate = 0.2
training_epochs = 15
batch_size = 2048
display_step = 1

# Create model

def multilayer_perceptron():
    model = Sequential()
    model.add(Reshape([28*28]))
    model.add(Linear(784, 256).set_name('mlp_fc1'))
    model.add(ReLU())
    model.add(Linear(256, 256).set_name('mlp_fc2'))
    model.add(ReLU())
    model.add(Linear(256, 10).set_name('mlp_fc3'))
    model.add(LogSoftMax())
    return model

model = multilayer_perceptron()

optimizer = Optimizer(
    model=model,
    training_rdd=train_data,
    criterion=ClassNLLCriterion(),
    optim_method=SGD(learningrate=learning_rate),
    end_trigger=MaxEpoch(training_epochs),
    batch_size=batch_size)

optimizer.set_validation(
    batch_size=batch_size,
    val_rdd=test_data,
    trigger=EveryEpoch(),
    val_method=[Top1Accuracy()]
)

app_name='multilayer_perceptron-'+dt.datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary = TrainSummary(log_dir='/tmp/bigdl_summaries',
                                     app_name=app_name)
train_summary.set_summary_trigger("Parameters", SeveralIteration(50))
val_summary = ValidationSummary(log_dir='/tmp/bigdl_summaries',
                                        app_name=app_name)
optimizer.set_train_summary(train_summary)
optimizer.set_val_summary(val_summary)
print "saving logs to ",app_name

%%time
# Boot training process
trained_model = optimizer.optimize()
print "Optimization Done."
