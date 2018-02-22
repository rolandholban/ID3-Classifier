import math, pprint, formatter

pp = pprint.PrettyPrinter(indent=2, width=80)

class Classifier:

    def __init__(self, D, attr_list = None):
        
        # If an attribute list is not given, gather the attributes.
        if attr_list is None:
            attr_list = self.preprocess(D)
            
        self.dt = self.generateDT(D, attr_list)

    def generateDT(self, D, attr_list):
        if self.classesSame(D):
            # If all classes are the same, return the class of
            #   the first tuple.
            return D[0][1]

        elif attr_list is None or len(attr_list) == 0:
            # If we have no attributes left, return the class
            #   that has the most occurrences
            return max(self.classList(D), key=self.classList(D).get)

        else:
            gain_composite = self.gain(D, attr_list)

            # Get the attribute with the highest gain
            split_attr = gain_composite[0]

            # Remove the split-attribute
            split_list = attr_list[:]
            split_list.remove(split_attr)


            # A dictionary to hold our recursive branches
            branches = {}

            # Iterate over possible values of our split-attribute
            for subset in gain_composite[1]:
                # The subsets should all have the same value
                #   for the split-attribute, so just use the first
                #   tuple's value as a key.
                branches[subset[0][0][split_attr]] = self.generateDT(subset, attr_list)

            branches["None"] = max(self.classList(D), key=self.classList(D).get)

            return (split_attr, branches)


    def information(self, classes):
        summation = 0
        num_tuples = sum(classes.values())
        for c, n in classes.items():
            summation += n/num_tuples * math.log(n/num_tuples, 2)
        return -summation

    def entropy(self, D, attribute):
        # Obtain all of the possible values for the given
        #   attribute.
        attr_values = []
        for entry in D:
            value = entry[0][attribute]
            if value not in attr_values:
                attr_values.append(value)

        # Build a list of data sets that all contain the same
        #   value for the given attribute.
        sub_d_list = []
        for value in attr_values:
            subset = []
            for entry in D:
                if entry[0][attribute] == value:
                    subset.append(entry)

            sub_d_list.append(subset)

        entropy_sum = 0
        for subset in sub_d_list:
            entropy_sum += len(subset) / len(D) * self.information(self.classList(subset))

        # Returns (Entropy, (split-attribute, list of subsets
        #   where the value for the split attribute is the same))
        return (entropy_sum, (attribute, sub_d_list))


    def gain(self, D, attr_list):
        gains = {}
        for attribute in attr_list:
            entropy_values = self.entropy(D, attribute)
            gains[attribute] = ((self.information(self.classList(D)) - entropy_values[0]), entropy_values[1])

        highest_gain = max(gains, key=gains.get)

        # We get the 1st element, because the 0th is not necessary
        #   now that we know we have the value with the max gain
        return gains[highest_gain][1]

    def classify(self, test_data):
        results = []
        for sample in test_data:
            results.append((sample, self.subClassifier(sample[0], self.dt)))
        return results

    def subClassifier(self, sample, dt):
        if type(dt) is tuple and dt[0] in sample and sample[dt[0]] in dt[1]:
            return self.subClassifier(sample, dt[1][sample[dt[0]]])
        else:
            if type(dt) is tuple:
                dt = dt[1]["None"]
            return dt

    def classList(self, D):
        # Takes a dataset and returns a dictionary containing
        #   each class, and how often it occurs
        classes = {}
        for c in D:
            if c[1] not in classes:
                classes[c[1]] = 1
            else:
                classes[c[1]] += 1

        return classes

    def preprocess(self, D):
        # Takes a dataset and returns a list containing every
        #   attribute
        attr_list = list(D[0][0].keys())
        for i in range(1, len(D)):
            for key in D[i][0].keys():
                if key not in attr_list:
                    attr_list.append(key)

        return attr_list

    def classesSame(self, D):
        # Takes a dataset and returns a boolean value based upon
        #   whether or not all of the classes in the set are the
        #   same.
        c = D[0][1]
        for i in range(1, len(D)):
            if D[i][1] != c:
                return False

        return True

def main(data):

    # Create our decision tree on the data given to us.
    # data[0] is the training data.
    classifier = Classifier(data[0])

    # Pretty print our decision tree
    pp.pprint(classifier.dt)

    print()

    # Get and print the results of running our decision tree
    #   on our training data. It is returned as a list.
    # Format of element in results: (<data>, <predicted_class>)
    results = classifier.classify(data[1])
    
    pp.pprint(results)

if __name__ == "__main__":
    main((formatter.format("data.csv"), formatter.format("test_data.csv")))