import tensorflow as tf
import tensorflow_decision_forests as tfdf


def randomforest():
    model = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)
    return model

def distributedgradient():
    model = tfdf.keras.DistributedGradientBoostedTreesModel(task = tfdf.keras.Task.REGRESSION)
    return model

def gradienttree():
    model = tfdf.keras.GradientBoostedTreesModel(task = tfdf.keras.Task.REGRESSION)
    return model