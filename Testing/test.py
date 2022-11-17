from softgym.utils.topology import RopeTopology
import pickle

objects = []
with open("./Datasets/single_step/Errors.pkl","rb") as f:
    while True:
        try:
            objects.append(pickle.load(f))
        except EOFError:
            break

for obj in objects:
    try:
        RopeTopology.from_geometry(obj)
    except:
        pass