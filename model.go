package prose

import (
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strings"
	
	"gonum.org/v1/gonum/mat"
)

// A Model holds the structures and data used internally by prose.
type Model struct {
	Name string

	tagger          *perceptronTagger
	extracter       *entityExtracter
	sentimentModel  *binaryMaxentClassifier
}

// DataSource provides training data to a Model.
type DataSource func(model *Model)

// UsingEntities creates a NER from labeled data.
func UsingEntities(data []EntityContext) DataSource {
	return UsingEntitiesAndTokenizer(data, NewIterTokenizer())
}

// UsingEntities creates a NER from labeled data and custom tokenizer.
func UsingEntitiesAndTokenizer(data []EntityContext, tokenizer Tokenizer) DataSource {
	return func(model *Model) {
		corpus := makeCorpus(data, model.tagger, tokenizer)
		model.extracter = extracterFromData(corpus)
	}
}

// SentimentTrainingData represents text with sentiment labels for training
type SentimentTrainingData struct {
	Text      string           // The text to analyze
	Label     SentimentClass   // The true sentiment label
	Language  Language         // The language of the text
}

// UsingSentiment creates a sentiment classifier from labeled data
func UsingSentiment(data []SentimentTrainingData) DataSource {
	return func(model *Model) {
		model.sentimentModel = sentimentModelFromData(data)
	}
}

// LabeledEntity represents an externally-labeled named-entity.
type LabeledEntity struct {
	Start int
	End   int
	Label string
}

// EntityContext represents text containing named-entities.
type EntityContext struct {
	// Is this is a correct entity?
	//
	// Some annotation software, e.g. Prodigy, include entities "rejected" by
	// its user. This allows us to handle those cases.
	Accept bool

	Spans []LabeledEntity // The entity locations relative to `Text`.
	Text  string          // The sentence containing the entities.
}

// ModelFromData creates a new Model from user-provided training data.
func ModelFromData(name string, sources ...DataSource) *Model {
	model := defaultModel(true, true)
	model.Name = name
	for _, source := range sources {
		source(model)
	}
	return model
}

// ModelFromDisk loads a Model from the user-provided location.
func ModelFromDisk(path string) *Model {
	filesys := os.DirFS(path)
	return &Model{
		Name: filepath.Base(path),

		extracter:      loadClassifier(filesys),
		tagger:         newPerceptronTagger(),
		sentimentModel: loadSentimentModel(filesys),
	}
}

// ModelFromFS loads a model from the
func ModelFromFS(name string, filesys fs.FS) *Model {
	// Locate a folder matching name within filesys
	var modelFS fs.FS
	err := fs.WalkDir(filesys, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		// Model located. Exit tree traversal
		if d.Name() == name {
			modelFS, err = fs.Sub(filesys, path)
			if err != nil {
				return err
			}
			return io.EOF
		}

		return nil
	})
	if err != io.EOF {
		checkError(err)
	}

	return &Model{
		Name: name,

		extracter:      loadClassifier(modelFS),
		tagger:         newPerceptronTagger(),
		sentimentModel: loadSentimentModel(modelFS),
	}
}

// Write saves a Model to the user-provided location.
func (m *Model) Write(path string) error {
	err := os.MkdirAll(path, os.ModePerm)
	// m.Tagger.model.Marshal(path)
	checkError(m.extracter.model.marshal(path))
	
	// Save sentiment model if available
	if m.sentimentModel != nil {
		sentimentPath := filepath.Join(path, "Sentiment")
		err = os.MkdirAll(sentimentPath, os.ModePerm)
		if err != nil {
			return err
		}
		checkError(m.sentimentModel.marshal(sentimentPath))
	}
	
	return err
}

// SentimentAnalyzer creates a sentiment analyzer from this model
func (m *Model) SentimentAnalyzer(lang Language, config SentimentConfig) *SentimentAnalyzer {
	if m.sentimentModel != nil {
		return NewSentimentAnalyzerFromModel(lang, config, m.sentimentModel)
	}
	// Fall back to basic analyzer if no trained model available
	return NewSentimentAnalyzer(lang, config)
}

/* TODO: External taggers
func loadTagger(path string) *perceptronTagger {
	var wts map[string]map[string]float64
	var tags map[string]string
	var classes []string

	loc := filepath.Join(path, "AveragedPerceptron")
	dec := getDiskAsset(filepath.Join(loc, "weights.gob"))
	checkError(dec.Decode(&wts))

	dec = getDiskAsset(filepath.Join(loc, "tags.gob"))
	checkError(dec.Decode(&tags))

	dec = getDiskAsset(filepath.Join(loc, "classes.gob"))
	checkError(dec.Decode(&classes))

	model := newAveragedPerceptron(wts, tags, classes)
	return newTrainedPerceptronTagger(model)
}*/

func loadClassifier(filesys fs.FS) *entityExtracter {
	var mapping map[string]int
	var weights []float64
	var labels []string

	maxent, err := fs.Sub(filesys, "Maxent")
	checkError(err)

	file, err := maxent.Open("mapping.gob")
	checkError(err)
	checkError(getDiskAsset(file).Decode(&mapping))

	file, err = maxent.Open("weights.gob")
	checkError(err)
	checkError(getDiskAsset(file).Decode(&weights))

	file, err = maxent.Open("labels.gob")
	checkError(err)
	checkError(getDiskAsset(file).Decode(&labels))

	model := newMaxentClassifier(weights, mapping, labels)
	return newTrainedEntityExtracter(model)
}

// loadSentimentModel loads a sentiment classification model from filesystem
func loadSentimentModel(filesys fs.FS) *binaryMaxentClassifier {
	var mapping map[string]int
	var weights []float64
	var labels []string

	sentiment, err := fs.Sub(filesys, "Sentiment")
	if err != nil {
		// Sentiment model not available
		return nil
	}

	// The marshal method creates a Maxent subdirectory
	maxent, err := fs.Sub(sentiment, "Maxent")
	if err != nil {
		return nil
	}

	file, err := maxent.Open("mapping.gob")
	if err != nil {
		return nil
	}
	defer file.Close()
	checkError(getDiskAsset(file).Decode(&mapping))

	file, err = maxent.Open("weights.gob")
	if err != nil {
		return nil
	}
	defer file.Close()
	checkError(getDiskAsset(file).Decode(&weights))

	file, err = maxent.Open("labels.gob")
	if err != nil {
		return nil
	}
	defer file.Close()
	checkError(getDiskAsset(file).Decode(&labels))

	return newMaxentClassifier(weights, mapping, labels)
}

// extractSimpleSentimentFeatures extracts simplified features for faster training
func extractSimpleSentimentFeatures(doc *Document) map[string]float64 {
	features := make(map[string]float64)
	tokens := doc.Tokens()
	
	// Simple unigram features (top 100 most important)
	wordCounts := make(map[string]int)
	for _, token := range tokens {
		word := strings.ToLower(token.Text)
		if len(word) > 2 { // Skip very short words
			wordCounts[word]++
		}
	}
	
	// Add word features
	for word, count := range wordCounts {
		if count > 0 {
			features["word:"+word] = float64(count)
		}
	}
	
	// Expanded intensity-aware sentiment indicators (must match sentiment.go)
	strongPosWords := []string{
		"amazing", "excellent", "outstanding", "brilliant", "perfect", "fantastic", "awesome", "superb", 
		"magnificent", "exceptional", "love", "adore", "incredible", "phenomenal", "extraordinary", 
		"spectacular", "wonderful", "marvelous", "fabulous", "sensational", "stunning", "breathtaking",
		"flawless", "impeccable", "divine", "sublime", "exquisite", "astounding", "remarkable", "thrilled",
		"ecstatic", "overjoyed", "delighted", "best", "greatest", "finest", "ultimate", "supreme"}
	
	weakPosWords := []string{
		"good", "nice", "okay", "fine", "decent", "adequate", "satisfactory", "reasonable", "solid",
		"fair", "acceptable", "pleasant", "comfortable", "suitable", "workable", "functional"}
	
	strongNegWords := []string{
		"terrible", "awful", "horrible", "disgusting", "atrocious", "appalling", "pathetic", "useless", 
		"worthless", "hate", "despise", "abysmal", "dreadful", "ghastly", "hideous", "revolting",
		"repulsive", "vile", "despicable", "detestable", "loathe", "abhor", "worst", "nightmare",
		"disaster", "catastrophe", "garbage", "trash", "junk", "crap", "shit", "hell", "damn"}
	
	weakNegWords := []string{
		"bad", "poor", "disappointing", "mediocre", "subpar", "inferior", "lacking", "flawed",
		"unsatisfactory", "inadequate", "unacceptable", "problematic", "deficient", "faulty"}
	
	strongPosCount := 0
	weakPosCount := 0
	strongNegCount := 0
	weakNegCount := 0
	
	for _, token := range tokens {
		word := strings.ToLower(token.Text)
		for _, w := range strongPosWords {
			if word == w {
				strongPosCount++
			}
		}
		for _, w := range weakPosWords {
			if word == w {
				weakPosCount++
			}
		}
		for _, w := range strongNegWords {
			if word == w {
				strongNegCount++
			}
		}
		for _, w := range weakNegWords {
			if word == w {
				weakNegCount++
			}
		}
	}
	
	features["strong_pos_count"] = float64(strongPosCount)
	features["weak_pos_count"] = float64(weakPosCount)
	features["strong_neg_count"] = float64(strongNegCount)
	features["weak_neg_count"] = float64(weakNegCount)
	
	// Overall sentiment polarity
	features["pos_count"] = float64(strongPosCount + weakPosCount)
	features["neg_count"] = float64(strongNegCount + weakNegCount)
	
	// Sentiment intensity ratio
	totalSentiment := float64(strongPosCount + weakPosCount + strongNegCount + weakNegCount)
	if totalSentiment > 0 {
		features["strong_sentiment_ratio"] = float64(strongPosCount + strongNegCount) / totalSentiment
	}
	
	// Enhanced distinguishing features for strong vs weak sentiment
	if strongNegCount > 0 && weakNegCount == 0 && strongPosCount == 0 {
		features["pure_strong_negative"] = 1.0
	}
	if strongPosCount > 0 && weakPosCount == 0 && strongNegCount == 0 {
		features["pure_strong_positive"] = 1.0
	}
	if weakNegCount > 0 && strongNegCount == 0 && strongPosCount == 0 {
		features["pure_weak_negative"] = 1.0
	}
	if weakPosCount > 0 && strongPosCount == 0 && strongNegCount == 0 {
		features["pure_weak_positive"] = 1.0
	}
	
	// Strong sentiment dominance patterns
	if strongNegCount > weakNegCount && strongNegCount > 0 {
		features["strong_neg_dominant"] = 1.0
	}
	if strongPosCount > weakPosCount && strongPosCount > 0 {
		features["strong_pos_dominant"] = 1.0
	}
	if weakNegCount > strongNegCount && weakNegCount > 0 {
		features["weak_neg_dominant"] = 1.0
	}
	if weakPosCount > strongPosCount && weakPosCount > 0 {
		features["weak_pos_dominant"] = 1.0
	}
	
	// Multiple strong words indicate very strong sentiment
	if strongNegCount > 1 {
		features["multiple_strong_negative"] = float64(strongNegCount)
	}
	if strongPosCount > 1 {
		features["multiple_strong_positive"] = float64(strongPosCount)
	}
	
	// Intensity ratios
	if strongNegCount + weakNegCount > 0 {
		features["strong_neg_ratio"] = float64(strongNegCount) / float64(strongNegCount + weakNegCount)
	}
	if strongPosCount + weakPosCount > 0 {
		features["strong_pos_ratio"] = float64(strongPosCount) / float64(strongPosCount + weakPosCount)
	}
	
	if strongPosCount == 0 && weakPosCount == 0 && strongNegCount == 0 && weakNegCount == 0 {
		features["no_sentiment_words"] = 1.0
	}
	
	// Length features
	features["length"] = float64(len(tokens))
	
	// Punctuation and intensity features
	exclamations := 0
	questions := 0
	allCaps := 0
	
	for _, token := range tokens {
		text := token.Text
		if strings.Contains(text, "!") {
			exclamations++
		}
		if strings.Contains(text, "?") {
			questions++
		}
		if len(text) > 2 && text == strings.ToUpper(text) && strings.ToLower(text) != text {
			allCaps++
		}
	}
	
	features["exclamations"] = float64(exclamations)
	features["questions"] = float64(questions)
	features["all_caps_words"] = float64(allCaps)
	
	// Enhanced intensity indicators
	multiExclamations := 0
	intensifiers := 0
	superlatives := 0
	
	intensifierWords := []string{"very", "extremely", "incredibly", "absolutely", "totally", "completely", 
		"utterly", "quite", "really", "truly", "highly", "deeply", "super", "so", "too", "way"}
	superlativeWords := []string{"most", "least", "best", "worst", "greatest", "smallest", "biggest", 
		"highest", "lowest", "finest", "ultimate", "maximum", "minimum"}
		
	for _, token := range tokens {
		text := strings.ToLower(token.Text)
		
		// Multiple exclamation marks
		if strings.Count(token.Text, "!") > 1 {
			multiExclamations++
		}
		
		// Intensifier words
		for _, intensifier := range intensifierWords {
			if text == intensifier {
				intensifiers++
				break
			}
		}
		
		// Superlative words
		for _, superlative := range superlativeWords {
			if text == superlative {
				superlatives++
				break
			}
		}
	}
	
	features["multi_exclamations"] = float64(multiExclamations)
	features["intensifiers"] = float64(intensifiers)
	features["superlatives"] = float64(superlatives)
	
	// Text pattern features for intensity
	fullText := ""
	for i, token := range tokens {
		if i > 0 {
			fullText += " "
		}
		fullText += token.Text
	}
	
	// Strong patterns that indicate intensity
	if strings.Contains(strings.ToUpper(fullText), "!!!") {
		features["triple_exclamation"] = 1.0
	}
	if strings.Contains(strings.ToUpper(fullText), "NEVER") ||
	   strings.Contains(strings.ToUpper(fullText), "ALWAYS") {
		features["absolute_language"] = 1.0
	}
	
	// Emotional expression patterns
	if strings.Contains(strings.ToLower(fullText), "can't believe") ||
	   strings.Contains(strings.ToLower(fullText), "cannot believe") {
		features["disbelief_expression"] = 1.0
	}
	
	// Count sentiment word density relative to text length
	if len(tokens) > 0 {
		sentimentDensity := totalSentiment / float64(len(tokens))
		features["sentiment_density"] = sentimentDensity
		
		if sentimentDensity > 0.3 {
			features["high_sentiment_density"] = 1.0
		}
	}
	
	return features
}

// empiricalCountSentiment calculates empirical feature counts for sentiment
func empiricalCountSentiment(corpus featureSet, encoding *binaryMaxentClassifier) *mat.VecDense {
	count := mat.NewVecDense(len(encoding.mapping)+1, nil)
	for _, entry := range corpus {
		for _, label := range encoding.labels {
			if label != entry.label {
				continue
			}
			// Encode features for this label
			for fname, fval := range entry.features {
				key := strings.Join([]string{fname, fval, label}, "-")
				if idx, found := encoding.mapping[key]; found {
					count.SetVec(idx, count.AtVec(idx)+1)
				}
			}
		}
	}
	return count
}

// expectedCountSentiment calculates expected feature counts under current model
func expectedCountSentiment(encoding *binaryMaxentClassifier, corpus featureSet) *mat.VecDense {
	count := mat.NewVecDense(len(encoding.mapping)+1, nil)
	
	for _, entry := range corpus {
		// Calculate probability distribution over labels
		probs := make(map[string]float64)
		sumExp := 0.0
		
		for _, label := range encoding.labels {
			score := 0.0
			for fname, fval := range entry.features {
				key := strings.Join([]string{fname, fval, label}, "-")
				if idx, found := encoding.mapping[key]; found {
					if idx < len(encoding.weights) && !math.IsInf(encoding.weights[idx], -1) {
						score += encoding.weights[idx]
					}
				}
			}
			expScore := math.Exp(score)
			probs[label] = expScore
			sumExp += expScore
		}
		
		// Normalize probabilities
		for label := range probs {
			probs[label] /= sumExp
		}
		
		// Add expected counts
		for _, label := range encoding.labels {
			prob := probs[label]
			for fname, fval := range entry.features {
				key := strings.Join([]string{fname, fval, label}, "-")
				if idx, found := encoding.mapping[key]; found {
					count.SetVec(idx, count.AtVec(idx)+prob)
				}
			}
		}
	}
	
	return count
}

// trainSentimentModel trains a sentiment model using GIS algorithm
func trainSentimentModel(corpus featureSet, featureKeys map[string]bool) *binaryMaxentClassifier {
	fmt.Printf("Training: Encoding features...\n")
	// First encode the features
	encoding := encodeSentiment(corpus, featureKeys)
	fmt.Printf("Training: Created encoding with %d features, %d labels\n", len(encoding.mapping), len(encoding.labels))
	
	// Add bias feature to each training example
	for i := range corpus {
		if corpus[i].features == nil {
			corpus[i].features = make(map[string]string)
		}
		corpus[i].features["__BIAS__"] = "1"
	}
	
	fmt.Printf("Training: Calculating empirical counts...\n")
	// Calculate empirical feature counts
	empCount := empiricalCountSentiment(corpus, encoding)
	rows, _ := empCount.Dims()
	
	// Find unattested features
	unattested := []int{}
	for index := 0; index < rows; index++ {
		if empCount.At(index, 0) == 0.0 {
			unattested = append(unattested, index)
		}
		// Take log of empirical counts
		if empCount.At(index, 0) > 0 {
			empCount.SetVec(index, math.Log(empCount.At(index, 0)))
		}
	}
	
	// Initialize weights
	weights := make([]float64, rows)
	for _, idx := range unattested {
		weights[idx] = math.Inf(-1)
	}
	encoding.weights = weights
	
	// Debug: Check label distribution in training
	labelCounts := make(map[string]int)
	for _, entry := range corpus {
		labelCounts[entry.label]++
	}
	// fmt.Printf("Training: Label distribution in corpus: %v\n", labelCounts) // Comment out for large datasets
	
	// GIS iterations
	fmt.Printf("Training: Starting GIS iterations...\n")
	cInv := 1.0 / float64(encoding.cardinality)
	fmt.Printf("Training: Cardinality factor cInv = %.6f\n", cInv)
	
	maxIter := 100 // Increase for better convergence
	
	for iter := 0; iter < maxIter; iter++ {
		if iter%20 == 0 {
			fmt.Printf("Training: Iteration %d/%d\n", iter+1, maxIter)
		}
		// Calculate expected counts
		estCount := expectedCountSentiment(encoding, corpus)
		
		// Add pseudocounts for unattested features
		for _, idx := range unattested {
			estCount.SetVec(idx, estCount.AtVec(idx)+1)
		}
		
		// Take log of expected counts
		for index := 0; index < rows; index++ {
			if estCount.At(index, 0) > 0 {
				estCount.SetVec(index, math.Log(estCount.At(index, 0)))
			}
		}
		
		// Update weights: w_i = w_i + (1/C) * (log(empirical) - log(expected))
		delta := mat.NewVecDense(rows, nil)
		delta.SubVec(empCount, estCount)
		delta.ScaleVec(cInv, delta)
		
		for index := 0; index < len(weights); index++ {
			if !math.IsInf(weights[index], -1) {
				weights[index] += delta.AtVec(index)
			}
		}
		
		encoding.weights = weights
		
		// Early stopping: check convergence every 10 iterations
		if iter > 0 && iter % 10 == 0 {
			// Simple convergence check - if weights aren't changing much, stop
			deltaSum := 0.0
			for index := 0; index < len(weights); index++ {
				if !math.IsInf(weights[index], -1) {
					deltaSum += math.Abs(delta.AtVec(index))
				}
			}
			avgDelta := deltaSum / float64(rows)
			if avgDelta < 0.0005 && iter > 30 { // Balanced convergence for large dataset
				fmt.Printf("Training: Converged at iteration %d (avg delta: %.6f)\n", iter+1, avgDelta)
				break
			}
			if iter%20 == 0 && iter > 0 {
				fmt.Printf("  Current avg delta: %.6f\n", avgDelta)
			}
		}
	}
	
	// Debug: Check final weight statistics  
	nonZeroWeights := 0
	infWeights := 0
	avgWeight := 0.0
	for _, w := range weights {
		if math.IsInf(w, -1) {
			infWeights++
		} else {
			nonZeroWeights++
			avgWeight += w
		}
	}
	if nonZeroWeights > 0 {
		avgWeight /= float64(nonZeroWeights)
	}
	fmt.Printf("Training: Final weights - %d finite (avg: %.4f), %d infinite\n", nonZeroWeights, avgWeight, infWeights)
	
	return encoding
}

// encodeSentiment creates a sentiment-specific encoded model
func encodeSentiment(corpus featureSet, featureKeys map[string]bool) *binaryMaxentClassifier {
	mapping := make(map[string]int) // maps (fname-fval-label) -> fid
	labels := []string{}
	
	// First pass: build mapping and collect labels
	for _, entry := range corpus {
		label := entry.label
		if !stringInSlice(label, labels) {
			labels = append(labels, label)
		}

		// Iterate through all features in the entry
		for fname, fval := range entry.features {
			entry := strings.Join([]string{fname, fval, label}, "-")
			if _, found := mapping[entry]; !found {
				mapping[entry] = len(mapping)
			}
		}
	}
	
	// Add correction feature for GIS
	for _, label := range labels {
		entry := strings.Join([]string{"__BIAS__", "1", label}, "-")
		if _, found := mapping[entry]; !found {
			mapping[entry] = len(mapping)
		}
	}
	
	// Calculate correct cardinality for GIS
	// Cardinality should be the maximum number of active features per example
	maxFeatureCount := 0
	for _, entry := range corpus {
		featureCount := len(entry.features) + 1 // +1 for bias
		if featureCount > maxFeatureCount {
			maxFeatureCount = featureCount
		}
	}
	
	// Initialize empty weights - will be filled by training
	weights := make([]float64, len(mapping))
	
	classifier := newMaxentClassifier(weights, mapping, labels)
	classifier.cardinality = maxFeatureCount  // Set correct cardinality
	fmt.Printf("Training: Set cardinality to %d (max features per example)\n", maxFeatureCount)
	return classifier
}

// sentimentModelFromData creates a sentiment model from training data
func sentimentModelFromData(data []SentimentTrainingData) *binaryMaxentClassifier {
	if len(data) == 0 {
		return nil
	}

	fmt.Printf("Building corpus from %d training examples...\n", len(data))
	// Create feature set for training
	corpus := make(featureSet, 0)
	labels := make(map[string]bool)
	allFeatureKeys := make(map[string]bool)
	
	for i, example := range data {
		if i%2000 == 0 {
			fmt.Printf("Processing example %d/%d\n", i+1, len(data))
		}
		// Create document and extract features
		doc, err := NewDocument(example.Text)
		if err != nil {
			continue
		}
		
		// Extract simplified features for faster training
		features := extractSimpleSentimentFeatures(doc)
		
		// Convert features to string map (required by existing training infrastructure)
		stringFeatures := make(map[string]string)
		for key, value := range features {
			stringFeatures[key] = fmt.Sprintf("%.6f", value)
			allFeatureKeys[key] = true
		}
		
		// Add to corpus
		label := string(example.Label)
		labels[label] = true
		corpus = append(corpus, feature{
			features: stringFeatures,
			label:    label,
		})
	}
	
	// Convert labels to slice
	labelSlice := make([]string, 0, len(labels))
	for label := range labels {
		labelSlice = append(labelSlice, label)
	}
	
	// Create and train model using sentiment-specific encoding
	model := trainSentimentModel(corpus, allFeatureKeys)
	
	return model
}

func defaultModel(tagging, classifying bool) *Model {
	var tagger *perceptronTagger
	var classifier *entityExtracter

	if tagging || classifying {
		tagger = newPerceptronTagger()
	}
	if classifying {
		classifier = newEntityExtracter()
	}

	return &Model{
		Name: "en-v2.0.0",

		tagger:         tagger,
		extracter:      classifier,
		sentimentModel: nil, // Will be loaded separately if needed
	}
}
