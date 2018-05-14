#import pre-processing.py
import crater_loader
import crater_deep_network
import matplotlib.pyplot as plt



print "Loading train, validation and test data"

training_data, validation_data, test_data = crater_loader.load_crater_data_phaseII_wrapper()

print "Training Network"

craternn = crater_deep_network.shallow(n=3, epochs=20)



nets = craternn
# plot the erroneous digits in the ensemble of nets just trained
error_locations, erroneous_predictions = ensemble(nets)
plt = plot_errors(error_locations, erroneous_predictions)
plt.savefig("ensemble_errors.png")
# plot the filters learned by the first of the nets just trained
plt = plot_filters(nets[0], 0, 5, 4)
plt.savefig("net_full_layer_0.png")
plt = plot_filters(nets[0], 1, 8, 5)
plt.savefig("net_full_layer_1.png")
