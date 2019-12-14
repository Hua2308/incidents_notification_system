import random
import collections
import math
import csv
import json

def generateClusteringExamples(numExamples, numWordsPerTopic, numFillerWords):
    '''
    Generate artificial examples inspired by sentiment for clustering.
    Each review has a hidden sentiment (positive or negative) and a topic (plot, acting, or music).
    The actual review consists of 2 sentiment words, 4 topic words and 2 filler words, for example:

        good:1 great:1 plot1:2 plot7:1 plot9:1 filler0:1 filler10:1

    numExamples: Number of examples to generate
    numWordsPerTopic: Number of words per topic (e.g., plot0, plot1, ...)
    numFillerWords: Number of words per filler (e.g., filler0, filler1, ...)
    '''
    sentiments = [['bad', 'awful', 'worst', 'terrible'], ['good', 'great', 'fantastic', 'excellent']]
    topics = ['plot', 'acting', 'music']
    def generateExample():
        x = collections.Counter()
        # Choose 2 sentiment words according to some sentiment
        sentimentWords = random.choice(sentiments)
        x[random.choice(sentimentWords)] += 1
        x[random.choice(sentimentWords)] += 1
        # Choose 4 topic words from a fixed topic
        topic = random.choice(topics)
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        x[topic + str(random.randint(0, numWordsPerTopic-1))] += 1
        # Choose 2 filler words
        x['filler' + str(random.randint(0, numFillerWords-1))] += 1
        return x

    random.seed(42)
    examples = [generateExample() for _ in range(numExamples)]
    return examples

def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in list(d2.items()))


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)

    def getDistance(pointA, pointB):
        vector = {}
        for dimension, value in pointA.items():
            vector[dimension] = (pointA[dimension] - pointB[dimension])

        # to simplify and have integer distance, do not apply square root to compute actual distance
        return math.sqrt(dotProduct(vector, vector))

    def getCentroid(cluster):
        vector = {}
        for point in cluster:
            for dimension, value in point.items():
                vector[dimension] = vector[dimension] + value if dimension in vector.keys() else value

        numberOfPoints = len(cluster)
        for dimension, value in vector.items():
            vector[dimension] = vector[dimension] / numberOfPoints
        return vector

    def getReconstructionLoss(list_of_clusters):

        reconstructionLoss = 0
        for centroid, cluster in list_of_clusters.items():
            for point in cluster:
                reconstructionLoss += getDistance(dict(centroid), point)
        return reconstructionLoss

    # Find the cluster for given point, and add point into it
    def addToCluster(point, list_of_clusters):
        min = None
        minCentroid = None
        for centroid, cluster in list_of_clusters.items():
            distance = getDistance(dict(centroid), point)
            if not min:
                min = distance
                minCentroid = dict(centroid)
            elif min and distance < min:
                min = distance
                minCentroid = dict(centroid)
        list_of_clusters[tuple(sorted(minCentroid.items()))].append(point)

    # Initialization with first K points
    list_of_clusters = dict()
    for i in range(0, K):
        cluster = list()
        cluster.append(examples[random.randint(0, len(examples))])
        # Centroid is key, cluster is value
        list_of_clusters[tuple(sorted(examples[i].items()))] = cluster
    print("initialization list of clusters with K points")
    print(list_of_clusters)
    # Clustering the rest of points
    for j in range(0, len(examples)):
        addToCluster(examples[j], list_of_clusters)
    print("Clustering the rest points in 1st iteration")
    print(list_of_clusters)
    # Iteration
    for k in range(2, maxIters + 2):

        print("iteration ", k)
        new_list_of_clusters = dict()
        for centroid, cluster in list_of_clusters.items():
            newCentroid = getCentroid(cluster)
            new_list_of_clusters[tuple(sorted(newCentroid.items()))] = list()

        for l in range(0, len(examples)):
            addToCluster(examples[l], new_list_of_clusters)
        list_of_clusters = new_list_of_clusters
        print(list_of_clusters)

    reconstructionLoss = getReconstructionLoss(list_of_clusters)
    print("reconstruction loss is ", reconstructionLoss)

    return K, list_of_clusters, reconstructionLoss
    # END_YOUR_CODE

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    counter = collections.Counter(x.split())
    return dict(counter)
    # END_YOUR_CODE

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)

    def calculateProjectedValue(feature):
        for word in feature:
            if word not in weights.keys():
                weights[word] = 0
        projectedValue = dotProduct(feature, weights)
        return projectedValue

    def calculateHingeLoss(projectedValue, trainExample):
        loss = max(0, (1 - projectedValue * trainExample[1]))
        return loss

    def predictor(example):
        if dotProduct(featureExtractor(example), weights) >=0:
            return 1
        else:
            return -1

    for i in range(0, numIters):
        for trainExample in trainExamples:
            feature = featureExtractor(trainExample[0])
            projectedValue = calculateProjectedValue(feature)
            loss = calculateHingeLoss(projectedValue, trainExample)
            if loss > 0:
                for word in feature:
                    weights[word] = weights[word] - eta * feature[word] * trainExample[1] * -1
        # print("Iteration {}: trainingExample loss {}, testExampleLoss {}".format(i,
        #                                                                          evaluatePredictor(trainExamples, predictor),
        #                                                                          evaluatePredictor(testExamples, predictor)))
    # END_YOUR_CODE
    return weights

def calculateProjectedValue(weights, feature):
    for word in feature:
        if word not in weights.keys():
            weights[word] = 0
    projectedValue = dotProduct(feature, weights)
    return projectedValue


############################  Variables  #########################################
hurricane_training_csv_file = 'hurricane_training.csv'
earthquake_training_csv_file = 'earthquake_training.csv'
training_number_of_iteration = 20
training_eta = 0.01
K = 2 # Initial K for K-Mean cluster
cluster_distance_threshold = 200 # centroids distance threshold to stop clustering
hurricane_testing_csv_file = 'hurricane_testing.csv'
earthquake_testing_csv_file = 'earthquake_testing.csv'
current_testing_file = hurricane_testing_csv_file

############################  Train models  #########################################

hurricane_weights = []
with open(hurricane_training_csv_file) as tweets:
    training_set = []
    csv_reader = csv.reader(tweets, delimiter=',')
    for row in csv_reader:
        tweet = row[2]
        label = row[6]
        if label and label != 'tag':
            training_set.append((tweet, int(label)))
    featureExtractor = extractWordFeatures
    hurricane_weights = learnPredictor(training_set, [], featureExtractor, numIters=training_number_of_iteration, eta=training_eta)

    results = []
    for word, value in hurricane_weights.items():
        if value > 0:
            results.append((value, word))

earthquake_weights = []
with open(earthquake_training_csv_file) as tweets:
    training_set = []
    csv_reader = csv.reader(tweets, delimiter=',')
    for row in csv_reader:
        tweet = row[1]
        label = row[0]
        if label and label != 'label':
            training_set.append((tweet, int(label)))
    featureExtractor = extractWordFeatures
    earthquake_weights = learnPredictor(training_set, [], featureExtractor, numIters=20, eta=0.01)

    results = []
    for word, value in earthquake_weights.items():
        if value > 0:
            results.append((value, word))

############################  K - Mean Cluster  #############################################

def calculateMinDistance(centroids):
    min_distance = 999999
    for centroid_a in centroids:
        for centroid_b in centroids:
            if centroid_a and centroid_b and centroid_a != centroid_b:
                distance = calculateDistance(centroid_a, centroid_b)
                if distance < min_distance:
                    min_distance = distance
    return min_distance


def calculateDistance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d

locations = list()
geo_tweet = dict()
with open(current_testing_file, encoding='utf-8') as tweets:
    csv_reader = csv.reader(tweets, delimiter=',')
    for row in csv_reader:
    # Column names are id, text, datetime, geo, predicted, tag
    # Get geo item
        geo_loc = row[3]
        print(geo_loc)
        # Skip None row and 1st header row:
        if geo_loc and geo_loc != "location":
            # Convert single quote to double quote
            geo_loc = geo_loc.replace("\'", "\"")
            # Convert str to dict
            geo_loc = json.loads(geo_loc)
            coordinates = geo_loc["coordinates"]
            locations.append({0: coordinates[0], 1: coordinates[1]})
            geo_tweet[(coordinates[0], coordinates[1])] = row[1]

    centroids = []
    while True:
        centers, assignments, totalCost = kmeans(locations, K, maxIters=10)
        print(assignments.keys())
        for centroid in assignments.keys():
            if centroid:
                centroids.append((centroid[0][1], centroid[1][1]))
        min_distance = calculateMinDistance(centroids)
        if min_distance < cluster_distance_threshold:
            break
        K += 1

    print(K)


################################   Classifier  #########################################

tweets_clusters = dict()
for center, tweet_list in assignments.items():
    tweets_cluster = []
    tweets_clusters[center] = []
    for tweet in tweet_list:
        tweets_cluster.append(geo_tweet[(tweet[0], tweet[1])])
    tweets_clusters[center].append(tweets_cluster)

for centroid, cluster in tweets_clusters.items():
    print("***************")
    total_tweets = len(cluster[0])
    hurricane_relevant = 0
    earthquake_relevant = 0
    for tweet in cluster[0]:
        hurricane_value = calculateProjectedValue(hurricane_weights, extractWordFeatures(tweet))
        earthquake_value = calculateProjectedValue(earthquake_weights, extractWordFeatures(tweet))
        if hurricane_value >= 1:
            hurricane_relevant += 1
        if earthquake_value >= 1:
            earthquake_relevant += 1

    print("Location: Latitude {}, Longitude {}".format(centroid[0][1], centroid[1][1]))
    print("Hurricane possibility: {}%".format(100 * hurricane_relevant/total_tweets))
    print("Earthquake possibility: {}%".format(100 * earthquake_relevant/total_tweets))
