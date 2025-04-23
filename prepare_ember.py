# save this as prepare_ember.py
import ember
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    ember.create_vectorized_features("./data/ember2018/")
    ember.create_metadata("./data/ember2018/")
    print("Vectorized features created!")