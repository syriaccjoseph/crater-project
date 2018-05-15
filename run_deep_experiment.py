#import pre-processing.py
import crater_loader
import crater_deep_network
import matplotlib.pyplot as plt



print "Loading train, validation and test data"

training_data, validation_data, test_data = crater_loader.load_crater_data_phaseII_wrapper()

print "Training Network"

craternn = crater_deep_network.shallow(n=3, epochs=20)


craterconvnn = crater_deep_network.our_conv_net(n=3, epochs=20)


nets = craterconvnn
# plot the erroneous digits in the ensemble of nets just trained
error_locations, erroneous_predictions = crater_deep_network.ensemble(nets)
plt = crater_deep_network.plot_errors(error_locations, erroneous_predictions)
plt.savefig("ensemble_errors.png")
# plot the filters learned by the first of the nets just trained
plt = crater_deep_network.plot_filters(nets[0], 0, 5, 4)
plt.savefig("net_full_layer_0.png")
plt = crater_deep_network.plot_filters(nets[0], 1, 8, 5)
plt.savefig("net_full_layer_1.png")
