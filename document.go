package prose

import (
	"context"
	"time"
)

// A DocOpt represents a setting that changes the document creation process.
//
// For example, it might disable named-entity extraction:
//
//    doc := prose.NewDocument("...", prose.WithExtraction(false))
type DocOpt func(doc *Document, opts *DocOpts)

// DocOpts controls the Document creation process:
type DocOpts struct {
	Extract         bool                     // If true, include named-entity extraction
	Segment         bool                     // If true, include segmentation
	Tag             bool                     // If true, include POS tagging
	Tokenizer       Tokenizer                // Tokenizer to use
	Context         context.Context          // Context for cancellation and timeouts
	Timeout         time.Duration            // Processing timeout
	TokenPool       *TokenPool               // Token pool for memory optimization
	ProgressCallback func(progress float64)  // Progress reporting callback
	Language        Language                 // Document language
}

// UsingTokenizer specifies the Tokenizer to use.
func UsingTokenizer(include Tokenizer) DocOpt {
	return func(doc *Document, opts *DocOpts) {
		// Tagging and entity extraction both require tokenization.
		opts.Tokenizer = include
	}
}

// WithTokenization can enable (the default) or disable tokenization.
// Deprecated: use UsingTokenizer instead.
func WithTokenization(include bool) DocOpt {
	return func(doc *Document, opts *DocOpts) {
		if !include {
			opts.Tokenizer = nil
		}
	}
}

// WithTagging can enable (the default) or disable POS tagging.
func WithTagging(include bool) DocOpt {
	return func(doc *Document, opts *DocOpts) {
		opts.Tag = include
	}
}

// WithSegmentation can enable (the default) or disable sentence segmentation.
func WithSegmentation(include bool) DocOpt {
	return func(doc *Document, opts *DocOpts) {
		opts.Segment = include
	}
}

// WithExtraction can enable (the default) or disable named-entity extraction.
func WithExtraction(include bool) DocOpt {
	return func(doc *Document, opts *DocOpts) {
		opts.Extract = include
	}
}

// UsingModel can enable (the default) or disable named-entity extraction.
func UsingModel(model *Model) DocOpt {
	return func(doc *Document, opts *DocOpts) {
		doc.Model = model
	}
}

// WithContext sets the context for document processing
func WithContext(ctx context.Context) DocOpt {
	return func(doc *Document, opts *DocOpts) {
		opts.Context = ctx
	}
}

// WithTimeout sets a timeout for document processing
func WithTimeout(timeout time.Duration) DocOpt {
	return func(doc *Document, opts *DocOpts) {
		opts.Timeout = timeout
	}
}

// WithTokenPool enables token pooling for memory optimization
func WithTokenPool(pool *TokenPool) DocOpt {
	return func(doc *Document, opts *DocOpts) {
		opts.TokenPool = pool
	}
}

// WithProgressCallback sets a progress reporting callback
func WithProgressCallback(callback func(float64)) DocOpt {
	return func(doc *Document, opts *DocOpts) {
		opts.ProgressCallback = callback
	}
}

// WithLanguage sets the document language
func WithLanguage(lang Language) DocOpt {
	return func(doc *Document, opts *DocOpts) {
		opts.Language = lang
	}
}

// A Document represents a parsed body of text.
type Document struct {
	Model    *Model
	Text     string
	Metadata DocumentMetadata

	entities  []Entity
	sentences []Sentence
	tokens    []*Token
}

// Tokens returns `doc`'s tokens.
func (doc *Document) Tokens() []Token {
	tokens := make([]Token, 0, len(doc.tokens))
	for _, tok := range doc.tokens {
		tokens = append(tokens, *tok)
	}
	return tokens
}

// Sentences returns `doc`'s sentences.
func (doc *Document) Sentences() []Sentence {
	return doc.sentences
}

// Entities returns `doc`'s entities.
func (doc *Document) Entities() []Entity {
	return doc.entities
}

var defaultOpts = DocOpts{
	Tokenizer: NewIterTokenizer(),
	Segment:   true,
	Tag:       true,
	Extract:   true,
	Context:   context.Background(),
	Timeout:   30 * time.Second,
	TokenPool: NewTokenPool(),
	Language:  English,
}

// NewDocument creates a Document according to the user-specified options.
//
// For example,
//
//    doc := prose.NewDocument("...")
func NewDocument(text string, opts ...DocOpt) (*Document, error) {
	startTime := time.Now()
	
	doc := Document{
		Text: text,
		Metadata: DocumentMetadata{
			ProcessedAt: startTime,
			Version:     "v3.0.0",
		},
	}
	
	base := defaultOpts
	for _, applyOpt := range opts {
		applyOpt(&doc, &base)
	}

	// Set up context with timeout
	ctx := base.Context
	if base.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, base.Timeout)
		defer cancel()
	}

	// Check for cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	if doc.Model == nil {
		doc.Model = defaultModel(base.Tag, base.Extract)
	}

	doc.Metadata.Language = base.Language
	
	// Progress reporting helper
	reportProgress := func(p float64) {
		if base.ProgressCallback != nil {
			base.ProgressCallback(p)
		}
	}

	// Segmentation
	if base.Segment {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		
		segmenter := newPunktSentenceTokenizer()
		doc.sentences = segmenter.segmentWithOffsets(text)
		doc.Metadata.SentenceCount = len(doc.sentences)
		reportProgress(0.25)
	}

	// Tokenization
	if base.Tokenizer != nil {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		
		doc.tokens = append(doc.tokens, base.Tokenizer.TokenizeWithOffsets(text)...)
		doc.Metadata.TokenCount = len(doc.tokens)
		reportProgress(0.5)
	}

	// POS Tagging
	if base.Tag || base.Extract {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		
		doc.tokens = doc.Model.tagger.tag(doc.tokens)
		reportProgress(0.75)
	}

	// Named Entity Recognition
	if base.Extract {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		
		doc.tokens = doc.Model.extracter.classify(doc.tokens)
		doc.entities = doc.Model.extracter.chunk(doc.tokens)
		doc.Metadata.EntityCount = len(doc.entities)
		reportProgress(1.0)
	}

	// Finalize metadata
	doc.Metadata.ProcessingTimeMs = time.Since(startTime).Milliseconds()
	
	return &doc, nil
}
