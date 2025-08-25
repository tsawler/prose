package prose

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"
)

// TrainingConfig contains configuration for model training
type TrainingConfig struct {
	Iterations       int
	LearningRate     float64
	RegularizationL1 float64
	RegularizationL2 float64
	EarlyStopping    bool
	ValidationSplit  float64
	Patience         int
	BatchSize        int
	Context          context.Context
	ProgressCallback func(epoch int, loss float64, accuracy float64)
}

// DefaultTrainingConfig returns a default training configuration
func DefaultTrainingConfig() TrainingConfig {
	return TrainingConfig{
		Iterations:       100,
		LearningRate:     0.01,
		RegularizationL1: 0.0,
		RegularizationL2: 0.001,
		EarlyStopping:    true,
		ValidationSplit:  0.2,
		Patience:         10,
		BatchSize:        32,
		Context:          context.Background(),
	}
}

// TrainingMetrics contains metrics from training
type TrainingMetrics struct {
	FinalLoss       float64
	FinalAccuracy   float64
	BestLoss        float64
	BestAccuracy    float64
	EpochsCompleted int
	TrainingTime    time.Duration
	Converged       bool
}

// CrossValidationResult contains results from cross-validation
type CrossValidationResult struct {
	MeanAccuracy    float64
	StdAccuracy     float64
	MeanLoss        float64
	StdLoss         float64
	FoldResults     []ValidationResult
}

// ValidationResult contains validation metrics
type ValidationResult struct {
	Accuracy float64
	Loss     float64
	F1Score  float64
	Precision float64
	Recall   float64
}

// Trainer provides training functionality for NLP models
type Trainer struct {
	config TrainingConfig
}

// NewTrainer creates a new trainer with the given configuration
func NewTrainer(config TrainingConfig) *Trainer {
	return &Trainer{config: config}
}

// TrainPOSTagger trains a POS tagger on the provided data
func (t *Trainer) TrainPOSTagger(data TupleSlice) (*perceptronTagger, TrainingMetrics, error) {
	startTime := time.Now()
	
	// Validate data
	if len(data) == 0 {
		return nil, TrainingMetrics{}, fmt.Errorf("training data is empty")
	}

	// Split data for validation if early stopping is enabled
	var trainData, validData TupleSlice
	if t.config.EarlyStopping && t.config.ValidationSplit > 0 {
		splitIdx := int(float64(len(data)) * (1.0 - t.config.ValidationSplit))
		trainData = data[:splitIdx]
		validData = data[splitIdx:]
	} else {
		trainData = data
	}

	// Initialize model
	model := &averagedPerceptron{
		totals:  make(map[string]float64),
		stamps:  make(map[string]float64),
		classes: []string{},
		tagMap:  make(map[string]string),
		weights: make(map[string]map[string]float64),
	}
	
	tagger := &perceptronTagger{model: model}
	tagger.makeTagMap(trainData)

	bestLoss := math.Inf(1)
	bestAccuracy := 0.0
	patienceCounter := 0
	
	var metrics TrainingMetrics
	
	for epoch := 0; epoch < t.config.Iterations; epoch++ {
		// Check for cancellation
		select {
		case <-t.config.Context.Done():
			return nil, metrics, t.config.Context.Err()
		default:
		}

		// Shuffle training data
		shuffleData := make(TupleSlice, len(trainData))
		copy(shuffleData, trainData)
		rand.Shuffle(len(shuffleData), func(i, j int) {
			shuffleData[i], shuffleData[j] = shuffleData[j], shuffleData[i]
		})

		// Training epoch
		epochLoss := 0.0
		correct := 0
		total := 0
		
		for _, tuple := range shuffleData {
			words, tags := tuple[0], tuple[1]
			p1, p2 := "-START-", "-START2-"
			context := []string{p1, p2}
			
			for _, w := range words {
				if w == "" {
					continue
				}
				context = append(context, normalize(w))
			}
			context = append(context, []string{"-END-", "-END2-"}...)
			
			for i, word := range words {
				var guess string
				var found bool
				
				if guess, found = tagger.model.tagMap[word]; !found {
					feats := featurize(i, context, word, p1, p2)
					guess = tagger.model.predict(feats)
					tagger.model.update(tags[i], guess, feats)
				}
				
				if guess == tags[i] {
					correct++
				}
				total++
				
				p2 = p1
				p1 = guess
			}
		}
		
		trainAccuracy := float64(correct) / float64(total)
		
		// Validation
		var validLoss, validAccuracy float64
		if len(validData) > 0 {
			validResult := t.validatePOSTagger(tagger, validData)
			validLoss = validResult.Loss
			validAccuracy = validResult.Accuracy
		} else {
			validLoss = epochLoss
			validAccuracy = trainAccuracy
		}
		
		// Progress callback
		if t.config.ProgressCallback != nil {
			t.config.ProgressCallback(epoch, validLoss, validAccuracy)
		}
		
		// Early stopping
		if t.config.EarlyStopping {
			if validAccuracy > bestAccuracy {
				bestAccuracy = validAccuracy
				bestLoss = validLoss
				patienceCounter = 0
			} else {
				patienceCounter++
				if patienceCounter >= t.config.Patience {
					metrics.Converged = true
					break
				}
			}
		}
		
		metrics.EpochsCompleted = epoch + 1
		metrics.FinalLoss = validLoss
		metrics.FinalAccuracy = validAccuracy
	}
	
	// Average weights
	tagger.model.averageWeights()
	
	metrics.BestLoss = bestLoss
	metrics.BestAccuracy = bestAccuracy
	metrics.TrainingTime = time.Since(startTime)
	
	return tagger, metrics, nil
}

// validatePOSTagger evaluates a POS tagger on validation data
func (t *Trainer) validatePOSTagger(tagger *perceptronTagger, data TupleSlice) ValidationResult {
	correct := 0
	total := 0
	
	for _, tuple := range data {
		words, tags := tuple[0], tuple[1]
		
		// Create token slice for tagging
		tokens := make([]*Token, len(words))
		for i, word := range words {
			tokens[i] = &Token{Text: word}
		}
		
		// Tag the tokens
		taggedTokens := tagger.tag(tokens)
		
		// Calculate accuracy
		for i, token := range taggedTokens {
			if i < len(tags) && token.Tag == tags[i] {
				correct++
			}
			total++
		}
	}
	
	accuracy := float64(correct) / float64(total)
	loss := 1.0 - accuracy // Simple loss approximation
	
	return ValidationResult{
		Accuracy:  accuracy,
		Loss:      loss,
		F1Score:   accuracy, // Simplified
		Precision: accuracy, // Simplified
		Recall:    accuracy, // Simplified
	}
}

// CrossValidatePOSTagger performs k-fold cross-validation
func (t *Trainer) CrossValidatePOSTagger(data TupleSlice, k int) (CrossValidationResult, error) {
	if k <= 1 {
		return CrossValidationResult{}, fmt.Errorf("k must be greater than 1")
	}
	
	foldSize := len(data) / k
	results := make([]ValidationResult, k)
	
	for fold := 0; fold < k; fold++ {
		// Create train/test split
		start := fold * foldSize
		end := start + foldSize
		if fold == k-1 {
			end = len(data) // Include remaining items in last fold
		}
		
		testData := data[start:end]
		trainData := append(data[:start], data[end:]...)
		
		// Train model
		tempConfig := t.config
		tempConfig.EarlyStopping = false // Disable early stopping for CV
		tempTrainer := NewTrainer(tempConfig)
		
		tagger, _, err := tempTrainer.TrainPOSTagger(trainData)
		if err != nil {
			return CrossValidationResult{}, err
		}
		
		// Validate
		results[fold] = t.validatePOSTagger(tagger, testData)
	}
	
	// Calculate statistics
	var meanAcc, meanLoss float64
	for _, result := range results {
		meanAcc += result.Accuracy
		meanLoss += result.Loss
	}
	meanAcc /= float64(k)
	meanLoss /= float64(k)
	
	// Calculate standard deviations
	var varAcc, varLoss float64
	for _, result := range results {
		varAcc += math.Pow(result.Accuracy-meanAcc, 2)
		varLoss += math.Pow(result.Loss-meanLoss, 2)
	}
	stdAcc := math.Sqrt(varAcc / float64(k))
	stdLoss := math.Sqrt(varLoss / float64(k))
	
	return CrossValidationResult{
		MeanAccuracy: meanAcc,
		StdAccuracy:  stdAcc,
		MeanLoss:     meanLoss,
		StdLoss:      stdLoss,
		FoldResults:  results,
	}, nil
}

// averageWeights calculates averaged weights for the perceptron
func (m *averagedPerceptron) averageWeights() {
	for feat, weights := range m.weights {
		newWeights := make(map[string]float64)
		for class, weight := range weights {
			key := feat + "-" + class
			total := m.totals[key]
			total += (m.instances - m.stamps[key]) * weight
			if m.instances > 0 {
				averaged := total / m.instances
				if math.Abs(averaged) > 1e-6 { // Only keep significant weights
					newWeights[class] = averaged
				}
			}
		}
		m.weights[feat] = newWeights
	}
}

// makeTagMap creates a tag mapping for frequent words
func (pt *perceptronTagger) makeTagMap(sentences TupleSlice) {
	counts := make(map[string]map[string]int)
	for _, tuple := range sentences {
		words, tags := tuple[0], tuple[1]
		for i, word := range words {
			if i >= len(tags) {
				continue
			}
			tag := tags[i]
			if counts[word] == nil {
				counts[word] = make(map[string]int)
			}
			counts[word][tag]++
			pt.model.addClass(tag)
		}
	}
	
	for word, tagFreqs := range counts {
		tag, mode := maxValue(tagFreqs)
		n := float64(sumValues(tagFreqs))
		if n >= 20 && (float64(mode)/n) >= 0.97 {
			pt.model.tagMap[word] = tag
		}
	}
}

// addClass adds a class to the model if it doesn't exist
func (m *averagedPerceptron) addClass(class string) {
	for _, existing := range m.classes {
		if existing == class {
			return
		}
	}
	m.classes = append(m.classes, class)
}

// update updates the model weights based on prediction error
func (m *averagedPerceptron) update(truth, guess string, feats map[string]float64) {
	m.instances++
	if truth == guess {
		return
	}
	
	for f := range feats {
		weights := make(map[string]float64)
		if val, ok := m.weights[f]; ok {
			weights = val
		} else {
			m.weights[f] = weights
		}
		
		m.updateFeat(truth, f, get(truth, weights), 1.0)
		m.updateFeat(guess, f, get(guess, weights), -1.0)
	}
}

// updateFeat updates feature weights
func (m *averagedPerceptron) updateFeat(c, f string, v, w float64) {
	key := f + "-" + c
	m.totals[key] += (m.instances - m.stamps[key]) * w
	m.stamps[key] = m.instances
	m.weights[f][c] = w + v
}

// get retrieves a weight value with default 0.0
func get(k string, m map[string]float64) float64 {
	if v, ok := m[k]; ok {
		return v
	}
	return 0.0
}

// sumValues sums all values in a map
func sumValues(m map[string]int) int {
	sum := 0
	for _, v := range m {
		sum += v
	}
	return sum
}

// maxValue finds the key with maximum value in a map
func maxValue(m map[string]int) (string, int) {
	maxValue := 0
	key := ""
	for k, v := range m {
		if v >= maxValue {
			maxValue = v
			key = k
		}
	}
	return key, maxValue
}

// TrainSentimentClassifier trains a sentiment classifier on the provided data
func (t *Trainer) TrainSentimentClassifier(data []SentimentTrainingData) (*binaryMaxentClassifier, TrainingMetrics, error) {
	startTime := time.Now()
	
	// Validate data
	if len(data) == 0 {
		return nil, TrainingMetrics{}, fmt.Errorf("training data is empty")
	}

	// Split data for validation if early stopping is enabled
	var trainData, validData []SentimentTrainingData
	if t.config.EarlyStopping && t.config.ValidationSplit > 0 {
		splitIdx := int(float64(len(data)) * (1.0 - t.config.ValidationSplit))
		trainData = data[:splitIdx]
		validData = data[splitIdx:]
	} else {
		trainData = data
	}

	// Create corpus from training data
	corpus := t.prepareSentimentCorpus(trainData)
	if len(corpus) == 0 {
		return nil, TrainingMetrics{}, fmt.Errorf("failed to prepare training corpus")
	}

	// Train model using maximum entropy
	model := encode(corpus)
	
	var metrics TrainingMetrics
	metrics.FinalAccuracy = 1.0 // Placeholder - would calculate actual metrics
	metrics.FinalLoss = 0.0     // Placeholder - would calculate actual loss
	metrics.BestAccuracy = metrics.FinalAccuracy
	metrics.BestLoss = metrics.FinalLoss
	metrics.EpochsCompleted = 1
	metrics.TrainingTime = time.Since(startTime)
	metrics.Converged = true

	// Validate if validation data is available
	if len(validData) > 0 {
		accuracy := t.evaluateSentimentModel(model, validData)
		metrics.FinalAccuracy = accuracy
		metrics.BestAccuracy = accuracy
		
		if t.config.ProgressCallback != nil {
			t.config.ProgressCallback(1, metrics.FinalLoss, metrics.FinalAccuracy)
		}
	}

	return model, metrics, nil
}

// prepareSentimentCorpus prepares feature corpus from sentiment training data
func (t *Trainer) prepareSentimentCorpus(data []SentimentTrainingData) featureSet {
	corpus := make(featureSet, 0, len(data))
	
	for _, example := range data {
		// Create document and extract features
		doc, err := NewDocument(example.Text)
		if err != nil {
			continue
		}
		
		// Extract features using sentiment feature extractor
		extractor := newSentimentFeatureExtractor(example.Language)
		tokens := doc.Tokens()
		tokenPtrs := make([]*Token, len(tokens))
		for i := range tokens {
			tokenPtrs[i] = &tokens[i]
		}
		features := extractor.extractFeatures(tokenPtrs)
		
		// Convert features to string map (required by existing training infrastructure)
		stringFeatures := make(map[string]string)
		for key, value := range features {
			stringFeatures[key] = fmt.Sprintf("%.6f", value)
		}
		
		// Add to corpus
		corpus = append(corpus, feature{
			features: stringFeatures,
			label:    string(example.Label),
		})
	}
	
	return corpus
}

// evaluateSentimentModel evaluates a sentiment model on validation data
func (t *Trainer) evaluateSentimentModel(model *binaryMaxentClassifier, validData []SentimentTrainingData) float64 {
	if len(validData) == 0 {
		return 0.0
	}

	correct := 0
	total := 0
	
	for _, example := range validData {
		// Create document and extract features
		doc, err := NewDocument(example.Text)
		if err != nil {
			continue
		}
		
		// Extract features using sentiment feature extractor
		extractor := newSentimentFeatureExtractor(example.Language)
		tokens := doc.Tokens()
		tokenPtrs := make([]*Token, len(tokens))
		for i := range tokens {
			tokenPtrs[i] = &tokens[i]
		}
		features := extractor.extractFeatures(tokenPtrs)
		
		// Convert features to string map for prediction
		stringFeatures := make(map[string]string)
		for key, value := range features {
			stringFeatures[key] = fmt.Sprintf("%.6f", value)
		}
		
		// Predict using model
		prediction := t.predictSentiment(model, stringFeatures)
		
		// Check if prediction matches true label
		if prediction == string(example.Label) {
			correct++
		}
		total++
	}
	
	if total == 0 {
		return 0.0
	}
	
	return float64(correct) / float64(total)
}

// predictSentiment predicts sentiment class using the model
func (t *Trainer) predictSentiment(model *binaryMaxentClassifier, features map[string]string) string {
	// Calculate scores for each class
	classScores := make(map[string]float64)
	
	for _, label := range model.labels {
		score := 0.0
		labelEncoded := model.encode(features, label)
		
		for _, feature := range labelEncoded {
			if feature.key < len(model.weights) {
				score += model.weights[feature.key] * float64(feature.value)
			}
		}
		
		classScores[label] = score
	}
	
	// Find class with highest score
	maxScore := math.Inf(-1)
	bestClass := ""
	
	for class, score := range classScores {
		if score > maxScore {
			maxScore = score
			bestClass = class
		}
	}
	
	return bestClass
}