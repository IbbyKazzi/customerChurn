import pickle
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def run():
  

