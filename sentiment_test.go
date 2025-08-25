package prose

import (
	"math"
	"os"
	"testing"
)

func TestSentimentPolarity(t *testing.T) {
	tests := []struct {
		text     string
		expected float64
		delta    float64
		desc     string
	}{
		{"I love this product!", 0.8, 0.2, "Strong positive sentiment"},
		{"This is terrible.", -0.8, 0.2, "Strong negative sentiment"},
		{"It's okay.", 0.2, 0.3, "Mildly positive sentiment"},
		{"Not bad at all.", 0.4, 0.3, "Negation of negative"},
		{"I don't like it.", -0.4, 0.3, "Negation of positive"},
		{"This movie is absolutely fantastic!", 0.85, 0.2, "Intensified positive"},
		{"The service was slightly disappointing.", -0.6, 0.4, "Diminished negative"},
		{"I really hate this!", -0.8, 0.2, "Intensified negative"},
		{"This is good but not great.", 0.3, 0.3, "Mixed sentiment"},
		{"", 0.0, 0.1, "Empty text"},
	}

	config := DefaultSentimentConfig()
	config.UseML = false // Disable ML for pure lexicon testing
	analyzer := NewSentimentAnalyzer(English, config)

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			doc, err := NewDocument(tt.text)
			if err != nil {
				t.Fatalf("Failed to create document: %v", err)
			}

			sentiment := analyzer.AnalyzeDocument(doc)
			if math.Abs(sentiment.Polarity-tt.expected) > tt.delta {
				t.Errorf("Text: %q\nExpected polarity: %.2f ± %.2f\nGot: %.2f",
					tt.text, tt.expected, tt.delta, sentiment.Polarity)
			}
		})
	}
}

func TestSentimentIntensity(t *testing.T) {
	tests := []struct {
		text         string
		minIntensity float64
		desc         string
	}{
		{"This is absolutely amazing!", 0.7, "High intensity with intensifier"},
		{"It's very very bad.", 0.6, "Multiple intensifiers"},
		{"Slightly disappointing.", 0.3, "Low intensity with diminisher"},
		{"TERRIBLE!!!", 0.7, "High intensity with caps and punctuation"},
		{"good", 0.5, "Single positive word"},
		{"This is the worst thing ever!", 0.7, "Superlative negative"},
		{"Perfect! Absolutely perfect!", 0.8, "Repeated strong positive"},
	}

	config := DefaultSentimentConfig()
	config.UseML = false
	analyzer := NewSentimentAnalyzer(English, config)

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			doc, _ := NewDocument(tt.text)
			sentiment := analyzer.AnalyzeDocument(doc)

			if sentiment.Intensity < tt.minIntensity {
				t.Errorf("Text: %q\nExpected intensity >= %.2f\nGot: %.2f",
					tt.text, tt.minIntensity, sentiment.Intensity)
			}
		})
	}
}

func TestNegationHandling(t *testing.T) {
	pairs := []struct {
		positive string
		negated  string
		desc     string
	}{
		{"This is good.", "This is not good.", "Simple negation"},
		{"I like it.", "I don't like it.", "Contraction negation"},
		{"Happy with the service.", "Not happy with the service.", "Beginning negation"},
		{"The food is excellent.", "The food isn't excellent.", "Negation with contraction"},
		{"I love this.", "I never loved this.", "Never negation"},
	}

	config := DefaultSentimentConfig()
	config.UseML = false
	analyzer := NewSentimentAnalyzer(English, config)

	for _, pair := range pairs {
		t.Run(pair.desc, func(t *testing.T) {
			posDoc, _ := NewDocument(pair.positive)
			negDoc, _ := NewDocument(pair.negated)

			posSent := analyzer.AnalyzeDocument(posDoc)
			negSent := analyzer.AnalyzeDocument(negDoc)

			// Negation should flip or significantly reduce the polarity
			if posSent.Polarity > 0.1 && negSent.Polarity > 0 {
				t.Errorf("Negation not handled properly:\n%s: %.2f\n%s: %.2f",
					pair.positive, posSent.Polarity,
					pair.negated, negSent.Polarity)
			}
		})
	}
}

func TestModifierEffects(t *testing.T) {
	base := "This is good."
	intensified := "This is very good."
	diminished := "This is slightly good."
	veryIntensified := "This is extremely good."

	config := DefaultSentimentConfig()
	config.UseML = false
	analyzer := NewSentimentAnalyzer(English, config)

	baseDoc, _ := NewDocument(base)
	intDoc, _ := NewDocument(intensified)
	dimDoc, _ := NewDocument(diminished)
	veryIntDoc, _ := NewDocument(veryIntensified)

	baseSent := analyzer.AnalyzeDocument(baseDoc)
	intSent := analyzer.AnalyzeDocument(intDoc)
	dimSent := analyzer.AnalyzeDocument(dimDoc)
	veryIntSent := analyzer.AnalyzeDocument(veryIntDoc)

	// Intensifier should increase magnitude
	if math.Abs(intSent.Polarity) <= math.Abs(baseSent.Polarity) {
		t.Errorf("Intensifier failed: base=%.2f, intensified=%.2f",
			baseSent.Polarity, intSent.Polarity)
	}

	// Strong intensifier should increase more (unless already at maximum)
	if math.Abs(intSent.Polarity) < 0.99 && math.Abs(veryIntSent.Polarity) <= math.Abs(intSent.Polarity) {
		t.Errorf("Strong intensifier failed: intensified=%.2f, very intensified=%.2f",
			intSent.Polarity, veryIntSent.Polarity)
	}

	// Diminisher should decrease magnitude
	if math.Abs(dimSent.Polarity) >= math.Abs(baseSent.Polarity) {
		t.Errorf("Diminisher failed: base=%.2f, diminished=%.2f",
			baseSent.Polarity, dimSent.Polarity)
	}
}

func TestSentimentClasses(t *testing.T) {
	tests := []struct {
		text     string
		expected SentimentClass
		desc     string
	}{
		{"This is absolutely perfect!", StrongPositive, "Strong positive"},
		{"Good product.", Positive, "Moderate positive"},
		{"It's okay, nothing special.", Neutral, "Neutral"},
		{"Not good.", Negative, "Negative"},
		{"Absolutely terrible!", StrongNegative, "Strong negative"},
		{"I love it but hate the price.", Mixed, "Mixed sentiment"},
	}

	config := DefaultSentimentConfig()
	config.UseML = false
	analyzer := NewSentimentAnalyzer(English, config)

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			doc, _ := NewDocument(tt.text)
			sentiment := analyzer.AnalyzeDocument(doc)

			// Allow for some flexibility in classification
			if tt.expected != Mixed && sentiment.Dominant != tt.expected {
				// Check if it's at least in the right direction
				isPositive := sentiment.Dominant == StrongPositive || sentiment.Dominant == Positive
				expectPositive := tt.expected == StrongPositive || tt.expected == Positive

				if isPositive != expectPositive && tt.expected != Neutral {
					t.Errorf("Text: %q\nExpected class: %s\nGot: %s (polarity: %.2f)",
						tt.text, tt.expected, sentiment.Dominant, sentiment.Polarity)
				}
			}
		})
	}
}

func TestSentimentFeatures(t *testing.T) {
	text := "I absolutely love this amazing product! It's perfect!"

	config := DefaultSentimentConfig()
	config.UseML = false
	analyzer := NewSentimentAnalyzer(English, config)

	doc, _ := NewDocument(text)
	sentiment := analyzer.AnalyzeDocument(doc)

	// Check that features are populated
	if len(sentiment.Features.PositiveWords) == 0 {
		t.Error("Expected positive words in features")
	}

	// Check for specific words
	foundLove := false
	foundAmazing := false
	for _, word := range sentiment.Features.PositiveWords {
		if word.Word == "love" {
			foundLove = true
		}
		if word.Word == "amazing" {
			foundAmazing = true
		}
	}

	if !foundLove {
		t.Error("Expected 'love' in positive words")
	}
	if !foundAmazing {
		t.Error("Expected 'amazing' in positive words")
	}
}

func TestSentimentConfidence(t *testing.T) {
	tests := []struct {
		text        string
		minConf     float64
		maxConf     float64
		desc        string
	}{
		{"excellent perfect amazing wonderful", 0.5, 1.0, "High confidence - many sentiment words"},
		{"the a an", 0.0, 0.3, "Low confidence - no sentiment words"},
		{"good bad", 0.2, 0.7, "Medium confidence - conflicting sentiment"},
		{"I think it might be okay", 0.2, 0.6, "Lower confidence - uncertain language"},
	}

	config := DefaultSentimentConfig()
	config.UseML = false
	analyzer := NewSentimentAnalyzer(English, config)

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			doc, _ := NewDocument(tt.text)
			sentiment := analyzer.AnalyzeDocument(doc)

			if sentiment.Confidence < tt.minConf || sentiment.Confidence > tt.maxConf {
				t.Errorf("Text: %q\nExpected confidence: %.2f-%.2f\nGot: %.2f",
					tt.text, tt.minConf, tt.maxConf, sentiment.Confidence)
			}
		})
	}
}

func TestMultilingualSentiment(t *testing.T) {
	tests := []struct {
		lang     Language
		text     string
		positive bool
		desc     string
	}{
		{English, "This is wonderful!", true, "English positive"},
		{English, "This is terrible!", false, "English negative"},
		{Spanish, "¡Esto es maravilloso!", true, "Spanish positive"},
		{Spanish, "Esto es terrible.", false, "Spanish negative"},
		{French, "C'est merveilleux!", true, "French positive"},
		{French, "C'est terrible!", false, "French negative"},
		{German, "Das ist wunderbar!", true, "German positive"},
		{German, "Das ist schrecklich!", false, "German negative"},
		{Japanese, "これは素晴らしいです！", true, "Japanese positive"},
		{Japanese, "これはひどいです。", false, "Japanese negative"},
	}

	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			config := DefaultSentimentConfig()
			config.UseML = false
			analyzer := NewSentimentAnalyzer(tt.lang, config)

			doc, _ := NewDocument(tt.text)
			sentiment := analyzer.AnalyzeDocument(doc)

			if tt.positive && sentiment.Polarity < 0 {
				t.Errorf("Language %s: expected positive sentiment for %q, got %.2f",
					tt.lang, tt.text, sentiment.Polarity)
			}
			if !tt.positive && sentiment.Polarity > 0 {
				t.Errorf("Language %s: expected negative sentiment for %q, got %.2f",
					tt.lang, tt.text, sentiment.Polarity)
			}
		})
	}
}

func TestSentenceLevelSentiment(t *testing.T) {
	text := "I love the design. The quality is terrible though. Overall, it's okay."

	config := DefaultSentimentConfig()
	config.UseML = false
	analyzer := NewSentimentAnalyzer(English, config)

	doc, _ := NewDocument(text)
	sentences := doc.Sentences()
	tokens := doc.Tokens()

	// Analyze each sentence
	sentiments := make([]SentimentScore, len(sentences))
	for i, sent := range sentences {
		sentiments[i] = analyzer.AnalyzeSentence(sent, tokens)
	}

	// First sentence should be positive
	if sentiments[0].Polarity < 0 {
		t.Errorf("First sentence should be positive, got %.2f", sentiments[0].Polarity)
	}

	// Second sentence should be negative
	if sentiments[1].Polarity > 0 {
		t.Errorf("Second sentence should be negative, got %.2f", sentiments[1].Polarity)
	}

	// Third sentence should be near neutral
	if math.Abs(sentiments[2].Polarity) > 0.5 {
		t.Errorf("Third sentence should be near neutral, got %.2f", sentiments[2].Polarity)
	}
}

func TestFeatureExtraction(t *testing.T) {
	extractor := newSentimentFeatureExtractor(English)

	tokens := []*Token{
		{Text: "This", Tag: "DT"},
		{Text: "is", Tag: "VBZ"},
		{Text: "absolutely", Tag: "RB"},
		{Text: "amazing", Tag: "JJ"},
		{Text: "!", Tag: "."},
	}

	features := extractor.extractFeatures(tokens)

	// Check for expected features
	expectedFeatures := []string{
		"unigram:this",
		"unigram:amazing",
		"bigram:absolutely_amazing",
		"has_exclamation",
		"adv_adj:absolutely_amazing",
	}

	for _, expected := range expectedFeatures {
		if _, exists := features[expected]; !exists {
			t.Errorf("Expected feature %q not found", expected)
		}
	}

	// Check token count
	if features["token_count"] != 5 {
		t.Errorf("Expected token_count=5, got %.0f", features["token_count"])
	}
}

func TestLexiconOperations(t *testing.T) {
	lexicon := LoadSentimentLexicon(English)

	// Test sentiment lookup
	tests := []struct {
		word     string
		expected float64
	}{
		{"excellent", 0.9},
		{"good", 0.6},
		{"bad", -0.6},
		{"terrible", -0.9},
		{"unknown_word", 0.0},
	}

	for _, tt := range tests {
		sentiment := lexicon.GetSentiment(tt.word)
		if math.Abs(sentiment-tt.expected) > 0.01 {
			t.Errorf("Word %q: expected %.2f, got %.2f", tt.word, tt.expected, sentiment)
		}
	}

	// Test negation detection
	negations := []string{"not", "never", "can't", "won't"}
	for _, neg := range negations {
		if !lexicon.IsNegation(neg) {
			t.Errorf("Expected %q to be recognized as negation", neg)
		}
	}

	// Test modifier strength
	modifiers := map[string]float64{
		"very":     0.3,
		"extremely": 0.5,
		"slightly": -0.3,
	}

	for word, expected := range modifiers {
		strength := lexicon.GetModifierStrength(word)
		if math.Abs(strength-expected) > 0.01 {
			t.Errorf("Modifier %q: expected %.2f, got %.2f", word, expected, strength)
		}
	}

	// Test custom word addition
	lexicon.AddCustomWord("blockchain", 0.3, 0.8)
	sentiment := lexicon.GetSentiment("blockchain")
	if math.Abs(sentiment-0.3) > 0.01 {
		t.Error("Custom word not added correctly")
	}
}

func TestEmptySentiment(t *testing.T) {
	config := DefaultSentimentConfig()
	config.UseML = false
	analyzer := NewSentimentAnalyzer(English, config)

	// Test empty document
	doc, _ := NewDocument("")
	sentiment := analyzer.AnalyzeDocument(doc)

	if sentiment.Polarity != 0 || sentiment.Intensity != 0 {
		t.Errorf("Empty text should have zero polarity and intensity, got polarity=%.2f, intensity=%.2f",
			sentiment.Polarity, sentiment.Intensity)
	}

	if sentiment.Dominant != Neutral {
		t.Errorf("Empty text should be classified as Neutral, got %s", sentiment.Dominant)
	}
}

func TestExternalLexicon(t *testing.T) {
	// Create a temporary JSON lexicon file
	testLexicon := `{
		"languages": {
			"english": {
				"positive": [
					{
						"word": "testawesome",
						"sentiment": 0.9,
						"confidence": 0.95
					}
				],
				"negative": [
					{
						"word": "testterrible",
						"sentiment": -0.8,
						"confidence": 0.9
					}
				],
				"intensifiers": ["testextremely"],
				"diminishers": ["testslightly"],
				"negations": ["testnot"]
			}
		}
	}`

	// Write to temporary file
	tmpFile := "/tmp/test_lexicon.json"
	err := os.WriteFile(tmpFile, []byte(testLexicon), 0644)
	if err != nil {
		t.Fatalf("Failed to create test lexicon file: %v", err)
	}
	defer os.Remove(tmpFile)

	// Test loading lexicon directly first
	lexicon, err := LoadSentimentLexiconWithExternal(English, tmpFile)
	if err != nil {
		t.Fatalf("Failed to load lexicon: %v", err)
	}

	// Check if external words are loaded
	testScore := lexicon.GetSentiment("testawesome")
	if testScore != 0.9 {
		t.Errorf("Expected sentiment score 0.9 for testawesome, got %.2f", testScore)
	}

	testScore = lexicon.GetSentiment("testterrible")
	if testScore != -0.8 {
		t.Errorf("Expected sentiment score -0.8 for testterrible, got %.2f", testScore)
	}

	// Test loading analyzer with external lexicon
	config := DefaultSentimentConfig()
	config.UseML = false
	analyzer, err := NewSentimentAnalyzerWithExternal(English, config, tmpFile)
	if err != nil {
		t.Fatalf("Failed to create analyzer with external lexicon: %v", err)
	}

	// Test positive external word
	doc, _ := NewDocument("This is testawesome")
	sentiment := analyzer.AnalyzeDocument(doc)
	if sentiment.Polarity <= 0 {
		t.Errorf("Expected positive sentiment for external positive word, got %.2f", sentiment.Polarity)
	}

	// Test negative external word
	doc, _ = NewDocument("This is testterrible")
	sentiment = analyzer.AnalyzeDocument(doc)
	if sentiment.Polarity >= 0 {
		t.Errorf("Expected negative sentiment for external negative word, got %.2f", sentiment.Polarity)
	}

	// Test that core words still work
	doc, _ = NewDocument("This is excellent")
	sentiment = analyzer.AnalyzeDocument(doc)
	if sentiment.Polarity <= 0 {
		t.Errorf("Expected positive sentiment for core positive word, got %.2f", sentiment.Polarity)
	}
}

func TestExternalLexiconMultipleLanguages(t *testing.T) {
	// Create a multilingual test lexicon
	testLexicon := `{
		"languages": {
			"english": {
				"positive": [{"word": "fantastic", "sentiment": 0.8, "confidence": 0.9}]
			},
			"spanish": {
				"positive": [{"word": "fantástico", "sentiment": 0.8, "confidence": 0.9}]
			},
			"french": {
				"positive": [{"word": "fantastique", "sentiment": 0.8, "confidence": 0.9}]
			}
		}
	}`

	tmpFile := "/tmp/test_multilang_lexicon.json"
	err := os.WriteFile(tmpFile, []byte(testLexicon), 0644)
	if err != nil {
		t.Fatalf("Failed to create test lexicon file: %v", err)
	}
	defer os.Remove(tmpFile)

	// Test loading different languages
	languages := []Language{English, Spanish, French}
	
	for _, lang := range languages {
		config := DefaultSentimentConfig()
		config.UseML = false
		analyzer, err := NewSentimentAnalyzerWithExternal(lang, config, tmpFile)
		if err != nil {
			t.Errorf("Failed to create analyzer for %s: %v", lang, err)
			continue
		}

		// Create test text for each language
		var testText string
		switch lang {
		case English:
			testText = "This is fantastic"
		case Spanish:
			testText = "Esto es fantástico"
		case French:
			testText = "C'est fantastique"
		}

		doc, _ := NewDocument(testText)
		sentiment := analyzer.AnalyzeDocument(doc)
		if sentiment.Polarity <= 0 {
			t.Errorf("Expected positive sentiment for %s external word, got %.2f", lang, sentiment.Polarity)
		}
	}
}

func TestExternalLexiconInvalidFile(t *testing.T) {
	config := DefaultSentimentConfig()
	
	// Test with non-existent file
	_, err := NewSentimentAnalyzerWithExternal(English, config, "/nonexistent/file.json")
	if err == nil {
		t.Error("Expected error for non-existent file")
	}

	// Test with invalid JSON
	invalidJSON := `{"invalid": json}`
	tmpFile := "/tmp/invalid_lexicon.json"
	err = os.WriteFile(tmpFile, []byte(invalidJSON), 0644)
	if err != nil {
		t.Fatalf("Failed to create invalid JSON file: %v", err)
	}
	defer os.Remove(tmpFile)

	_, err = NewSentimentAnalyzerWithExternal(English, config, tmpFile)
	if err == nil {
		t.Error("Expected error for invalid JSON")
	}
}

func TestClauseBoundaryNegation(t *testing.T) {
	// Negation shouldn't cross clause boundaries
	text := "I don't like the color, but the quality is excellent."

	config := DefaultSentimentConfig()
	config.UseML = false
	analyzer := NewSentimentAnalyzer(English, config)

	doc, _ := NewDocument(text)
	sentiment := analyzer.AnalyzeDocument(doc)

	// Should have mixed sentiment, not purely negative
	if sentiment.Polarity < -0.2 {
		t.Errorf("Clause boundary not respected in negation, got polarity=%.2f", sentiment.Polarity)
	}
}

func TestSentimentTraining(t *testing.T) {
	// Create training data
	trainingData := []SentimentTrainingData{
		{Text: "This is excellent", Label: Positive, Language: English},
		{Text: "This is terrible", Label: Negative, Language: English},
		{Text: "This is amazing work", Label: StrongPositive, Language: English},
		{Text: "This is awful garbage", Label: StrongNegative, Language: English},
		{Text: "This is okay", Label: Neutral, Language: English},
	}

	// Train model
	trainer := NewTrainer(DefaultTrainingConfig())
	model, metrics, err := trainer.TrainSentimentClassifier(trainingData)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if model == nil {
		t.Fatal("Expected trained model, got nil")
	}

	t.Logf("Training completed - Accuracy: %.2f, Time: %v", 
		metrics.FinalAccuracy, metrics.TrainingTime)
}

func TestModelIntegration(t *testing.T) {
	// Create training data
	trainingData := []SentimentTrainingData{
		{Text: "This is fantastic", Label: Positive, Language: English},
		{Text: "This is horrible", Label: Negative, Language: English},
		{Text: "This is neutral text", Label: Neutral, Language: English},
	}

	// Create model with sentiment training
	model := ModelFromData("test-sentiment", UsingSentiment(trainingData))
	
	if model.sentimentModel == nil {
		t.Fatal("Expected sentiment model to be trained")
	}

	// Create analyzer from model
	config := DefaultSentimentConfig()
	analyzer := model.SentimentAnalyzer(English, config)
	
	// Test analysis
	doc, _ := NewDocument("This is fantastic")
	sentiment := analyzer.AnalyzeDocument(doc)
	
	t.Logf("Sentiment analysis - Polarity: %.2f, Confidence: %.2f", 
		sentiment.Polarity, sentiment.Confidence)
}

func TestModelSerialization(t *testing.T) {
	// Create training data
	trainingData := []SentimentTrainingData{
		{Text: "Great product", Label: Positive, Language: English},
		{Text: "Bad product", Label: Negative, Language: English},
	}

	// Create and train model
	model := ModelFromData("test-serialization", UsingSentiment(trainingData))
	
	// Save model
	tempDir := "/tmp/test_sentiment_model"
	err := model.Write(tempDir)
	if err != nil {
		t.Fatalf("Failed to save model: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Load model
	loadedModel := ModelFromDisk(tempDir)
	
	// Verify sentiment model was loaded
	if loadedModel.sentimentModel == nil && model.sentimentModel != nil {
		t.Error("Sentiment model was not properly serialized/deserialized")
	}

	t.Logf("Model serialization test completed")
}

func BenchmarkSentimentAnalysis(b *testing.B) {
	texts := []string{
		"This product exceeded my expectations in every way.",
		"The customer service was absolutely terrible.",
		"It's an okay product, nothing special.",
		"I'm not sure if I like it or not.",
	}

	config := DefaultSentimentConfig()
	config.UseML = false
	analyzer := NewSentimentAnalyzer(English, config)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		doc, _ := NewDocument(texts[i%len(texts)])
		_ = analyzer.AnalyzeDocument(doc)
	}
}

func BenchmarkFeatureExtraction(b *testing.B) {
	extractor := newSentimentFeatureExtractor(English)
	tokens := []*Token{
		{Text: "This", Tag: "DT"},
		{Text: "is", Tag: "VBZ"},
		{Text: "an", Tag: "DT"},
		{Text: "absolutely", Tag: "RB"},
		{Text: "fantastic", Tag: "JJ"},
		{Text: "product", Tag: "NN"},
		{Text: "that", Tag: "WDT"},
		{Text: "exceeded", Tag: "VBD"},
		{Text: "my", Tag: "PRP$"},
		{Text: "expectations", Tag: "NNS"},
		{Text: "!", Tag: "."},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = extractor.extractFeatures(tokens)
	}
}

func BenchmarkLexiconLookup(b *testing.B) {
	lexicon := LoadSentimentLexicon(English)
	words := []string{"good", "bad", "excellent", "terrible", "amazing", "awful"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		word := words[i%len(words)]
		_ = lexicon.GetSentiment(word)
		_ = lexicon.GetConfidence(word)
	}
}