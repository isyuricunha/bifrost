package schemas

import (
	"fmt"
)

type BifrostEmbeddingRequest struct {
	Provider       ModelProvider        `json:"provider"`
	Model          string               `json:"model"`
	Input          *EmbeddingInput      `json:"input,omitempty"`
	Params         *EmbeddingParameters `json:"params,omitempty"`
	Fallbacks      []Fallback           `json:"fallbacks,omitempty"`
	RawRequestBody []byte               `json:"-"` // set bifrost-use-raw-request-body to true in ctx to use the raw request body. Bifrost will directly send this to the downstream provider.
}

func (r *BifrostEmbeddingRequest) GetRawRequestBody() []byte {
	return r.RawRequestBody
}

type BifrostEmbeddingResponse struct {
	Data        []EmbeddingData            `json:"data"` // Maps to "data" field in provider responses (e.g., OpenAI embedding format)
	Model       string                     `json:"model"`
	Object      string                     `json:"object"` // "list"
	Usage       *BifrostLLMUsage           `json:"usage"`
	ExtraFields BifrostResponseExtraFields `json:"extra_fields"`
}

// EmbeddingInput represents the input for an embedding request.
type EmbeddingInput struct {
	Contents []EmbeddingContent
}

type EmbeddingContent []EmbeddingContentPart

type EmbeddingContentPartType string

const (
	EmbeddingContentPartTypeText   EmbeddingContentPartType = "text"
	EmbeddingContentPartTypeImage  EmbeddingContentPartType = "image"
	EmbeddingContentPartTypeAudio  EmbeddingContentPartType = "audio"
	EmbeddingContentPartTypeFile   EmbeddingContentPartType = "file"
	EmbeddingContentPartTypeVideo  EmbeddingContentPartType = "video"
	EmbeddingContentPartTypeTokens EmbeddingContentPartType = "tokens"
)

type EmbeddingContentPart struct {
	Type EmbeddingContentPartType `json:"type"`

	Text   *string             `json:"text,omitempty"`
	Image  *EmbeddingMediaPart `json:"image,omitempty"`
	Audio  *EmbeddingMediaPart `json:"audio,omitempty"`
	File   *EmbeddingMediaPart `json:"file,omitempty"`
	Video  *EmbeddingMediaPart `json:"video,omitempty"`
	Tokens []int               `json:"tokens,omitempty"`
}

type EmbeddingMediaPart struct {
	Data     *string                `json:"data,omitempty"`
	URL      *string                `json:"url,omitempty"`
	MIMEType *string                `json:"mime_type,omitempty"`
	Filename *string                `json:"filename,omitempty"`
	Detail   *string                `json:"detail,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

func (m *EmbeddingMediaPart) Validate() error {
	if m == nil {
		return fmt.Errorf("embedding media payload is nil")
	}
	set := 0
	if m.Data != nil {
		set++
	}
	if m.URL != nil {
		set++
	}
	if set != 1 {
		return fmt.Errorf("embedding media payload must set exactly one of data or url")
	}
	return nil
}

func (p EmbeddingContentPart) Validate() error {
	set := 0
	if p.Text != nil {
		set++
	}
	if p.Image != nil {
		set++
	}
	if p.Audio != nil {
		set++
	}
	if p.File != nil {
		set++
	}
	if p.Video != nil {
		set++
	}
	if p.Tokens != nil {
		set++
	}
	if set != 1 {
		return fmt.Errorf("embedding content part must set exactly one modality")
	}

	switch p.Type {
	case EmbeddingContentPartTypeText:
		if p.Text == nil {
			return fmt.Errorf("embedding content part type %q requires text payload", p.Type)
		}
	case EmbeddingContentPartTypeImage:
		if p.Image == nil {
			return fmt.Errorf("embedding content part type %q requires image payload", p.Type)
		}
		return p.Image.Validate()
	case EmbeddingContentPartTypeAudio:
		if p.Audio == nil {
			return fmt.Errorf("embedding content part type %q requires audio payload", p.Type)
		}
		return p.Audio.Validate()
	case EmbeddingContentPartTypeFile:
		if p.File == nil {
			return fmt.Errorf("embedding content part type %q requires file payload", p.Type)
		}
		return p.File.Validate()
	case EmbeddingContentPartTypeVideo:
		if p.Video == nil {
			return fmt.Errorf("embedding content part type %q requires video payload", p.Type)
		}
		return p.Video.Validate()
	case EmbeddingContentPartTypeTokens:
		if p.Tokens == nil {
			return fmt.Errorf("embedding content part type %q requires tokens payload", p.Type)
		}
	default:
		return fmt.Errorf("unsupported embedding content part type %q", p.Type)
	}

	return nil
}

func (c EmbeddingContent) Validate() error {
	if len(c) == 0 {
		return fmt.Errorf("embedding content is empty")
	}
	for _, part := range c {
		if err := part.Validate(); err != nil {
			return err
		}
	}
	return nil
}

func (e *EmbeddingInput) Validate() error {
	if e == nil || len(e.Contents) == 0 {
		return fmt.Errorf("embedding input is empty")
	}
	for _, content := range e.Contents {
		if err := content.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// IsSingleContent returns true when the input represents a single embedding (one content entry).
func (e *EmbeddingInput) IsSingleContent() bool {
	return e != nil && len(e.Contents) == 1
}

// IsBatchContent returns true when the input represents multiple embeddings.
func (e *EmbeddingInput) IsBatchContent() bool {
	return e != nil && len(e.Contents) > 1
}

func (e *EmbeddingInput) HasAnyValue() bool {
	return e != nil && len(e.Contents) > 0
}

// GetContents returns the contents slice directly.
func (e *EmbeddingInput) GetContents() []EmbeddingContent {
	if e == nil {
		return nil
	}
	return e.Contents
}

type EmbeddingParameters struct {
	EncodingFormat *string `json:"encoding_format,omitempty"` // Format for embedding output (e.g., "float", "base64")
	Dimensions     *int    `json:"dimensions,omitempty"`      // Number of dimensions for embedding output
	TaskType       *string `json:"task_type,omitempty"`       // Intended embedding task
	Title          *string `json:"title,omitempty"`           // Optional title for the content
	AutoTruncate   *bool   `json:"auto_truncate,omitempty"`   // Automatically truncate long inputs
	Truncate       *string `json:"truncate,omitempty"`        // Provider-specific truncation strategy
	MaxTokens      *int    `json:"max_tokens,omitempty"`      // Maximum tokens to process

	// Dynamic parameters that can be provider-specific, they are directly
	// added to the request as is.
	ExtraParams map[string]interface{} `json:"-"`
}

type EmbeddingData struct {
	Index     int              `json:"index"`
	Object    string           `json:"object"`    // "embedding"
	Embedding EmbeddingsByType `json:"embedding"` // can be string, []float64, [][]float64, []int8, or []int32
}

type EmbeddingsByType struct {
	Float   []float64 `json:"float,omitempty"`   // Float embeddings
	Int8    []int8    `json:"int8,omitempty"`    // Int8 embeddings
	Uint8   []uint8   `json:"uint8,omitempty"`   // Uint8 embeddings
	Binary  []int8    `json:"binary,omitempty"`  // Binary embeddings
	Ubinary []uint8   `json:"ubinary,omitempty"` // Unsigned binary embeddings
	Base64  *string   `json:"base64,omitempty"`  // Base64 embeddings
}
