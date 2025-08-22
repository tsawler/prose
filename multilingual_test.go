package prose

import (
	"testing"
)

func TestGetStopWords(t *testing.T) {
	tests := []struct {
		name     string
		language Language
		expected []string // Sample of expected stop words
	}{
		{
			name:     "English stop words",
			language: English,
			expected: []string{"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"},
		},
		{
			name:     "Spanish stop words",
			language: Spanish,
			expected: []string{"el", "la", "de", "que", "y", "a", "en", "un", "por"},
		},
		{
			name:     "French stop words",
			language: French,
			expected: []string{"le", "de", "un", "et", "être", "avoir", "que", "pour", "dans"},
		},
		{
			name:     "German stop words",
			language: German,
			expected: []string{"der", "die", "und", "in", "den", "von", "zu", "das", "mit"},
		},
		{
			name:     "Japanese stop words",
			language: Japanese,
			expected: []string{"の", "は", "を", "に", "が", "と", "で", "て"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			processor := NewLanguageSpecificProcessor(tt.language)
			stopWords := processor.GetStopWords()

			// Check that we got some stop words
			if len(stopWords) == 0 {
				t.Errorf("Expected stop words for %s, but got none", tt.language)
			}

			// Create a map for easier lookup
			stopWordsMap := make(map[string]bool)
			for _, word := range stopWords {
				stopWordsMap[word] = true
			}

			// Check that expected stop words are present
			missing := []string{}
			for _, expected := range tt.expected {
				if !stopWordsMap[expected] {
					missing = append(missing, expected)
				}
			}

			if len(missing) > 0 {
				t.Errorf("Missing expected stop words for %s: %v\nGot %d stop words total", 
					tt.language, missing, len(stopWords))
			}

			// Log the count for information
			t.Logf("%s: Found %d stop words", tt.language, len(stopWords))
		})
	}
}

func TestMultilingualDocumentStopWords(t *testing.T) {
	// Test that multilingual document can access stop words
	texts := map[Language]string{
		English: "The quick brown fox jumps over the lazy dog",
		Spanish: "El rápido zorro marrón salta sobre el perro perezoso",
		French:  "Le renard brun rapide saute par-dessus le chien paresseux",
		German:  "Der schnelle braune Fuchs springt über den faulen Hund",
	}

	for lang, text := range texts {
		doc, err := NewMultilingualDocument(text)
		if err != nil {
			t.Errorf("Failed to create multilingual document for %s: %v", lang, err)
			continue
		}

		stopWords := doc.GetStopWords()
		if len(stopWords) == 0 {
			t.Errorf("Expected stop words for %s, but got none", lang)
		}

		t.Logf("Expected: %s, Detected: %s (confidence: %.2f), Stop words count: %d",
			lang, doc.detectedLanguage, doc.confidence, len(stopWords))
	}
}

func TestStopWordsLibraryIntegration(t *testing.T) {
	// Test that the library actually filters stop words as expected
	testCases := []struct {
		language Language
		text     string
		word     string
		isStop   bool
	}{
		{English, "the", "the", true},
		{English, "programming", "programming", false},
		{Spanish, "el", "el", true},
		{Spanish, "programación", "programación", false},
		{French, "le", "le", true},
		{French, "programmation", "programmation", false},
		{German, "der", "der", true},
		{German, "programmierung", "programmierung", false},
	}

	for _, tc := range testCases {
		processor := NewLanguageSpecificProcessor(tc.language)
		stopWords := processor.GetStopWords()
		
		stopWordsMap := make(map[string]bool)
		for _, word := range stopWords {
			stopWordsMap[word] = true
		}

		isStopWord := stopWordsMap[tc.word]
		if isStopWord != tc.isStop {
			t.Errorf("Language %s: expected '%s' stop word status to be %v, got %v",
				tc.language, tc.word, tc.isStop, isStopWord)
		}
	}
}