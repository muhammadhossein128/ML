class FindS:
    def __init__(self, num_attributes):
        # Initialize with the most specific hypothesis
        self.num_attributes = num_attributes
        self.hypothesis = ['0'] * self.num_attributes

    def is_consistent(self, instance, hypothesis=None):
        # Check if a hypothesis covers the given instance
        target_h = self.hypothesis if hypothesis is None else hypothesis
        for h_val, x_val in zip(target_h, instance):
            if h_val == '0':
                return False
            if h_val != '?' and h_val != x_val:
                return False
        return True

    def more_general_than(self, h1, h2):
        # Check if h1 is more general than or equal to h2
        for v1, v2 in zip(h1, h2):
            if v1 == '0' and v2 != '0':
                return False
            if v1 != '?' and v1 != v2 and v2 != '0':
                return False
        return True

    def fit_step(self, instance, label):
        # Update current hypothesis using a single training example
        if label != 'Yes':
            return list(self.hypothesis)

        for i in range(self.num_attributes):
            if self.hypothesis[i] == '0':
                self.hypothesis[i] = instance[i]
            elif self.hypothesis[i] != instance[i]:
                self.hypothesis[i] = '?'
        return list(self.hypothesis)

    def fit_with_trace(self, dataset):
        # Train on a full dataset and record hypothesis after each example
        trace = [list(self.hypothesis)]
        for instance, label in dataset:
            self.fit_step(instance, label)
            trace.append(list(self.hypothesis))
        return trace

