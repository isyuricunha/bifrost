package vertex

import (
	"fmt"
	"strings"

	"github.com/maximhq/bifrost/core/providers/gemini"
	"github.com/maximhq/bifrost/core/schemas"
)

// isVertexNativeMultimodalEmbeddingModel returns true for the Vertex-native
// multimodal embedding model (multimodalembedding@001). This model uses the
// :predict endpoint but with a different instance format (text/image/video fields
// instead of content) and a different response format (textEmbedding/imageEmbedding).
func isVertexNativeMultimodalEmbeddingModel(model string) bool {
	return strings.Contains(strings.ToLower(strings.TrimSpace(model)), "multimodalembedding")
}

func isVertexGeminiEmbeddingModel(model string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	return strings.Contains(model, "gemini-embedding-2")
}

// ToVertexEmbeddingRequest converts a Bifrost embedding request to Vertex AI text embedding format.
// All contents must be text-only. Each content entry maps to one instance (one output embedding).
func ToVertexEmbeddingRequest(bifrostReq *schemas.BifrostEmbeddingRequest) *VertexEmbeddingRequest {
	if bifrostReq == nil || bifrostReq.Input == nil || len(bifrostReq.Input.Contents) == 0 {
		return nil
	}

	vertexReq := &VertexEmbeddingRequest{}
	if bifrostReq.Params != nil {
		vertexReq.ExtraParams = bifrostReq.Params.ExtraParams
	}

	instances := make([]VertexEmbeddingInstance, 0, len(bifrostReq.Input.Contents))
	for _, content := range bifrostReq.Input.Contents {
		// Vertex text embedding expects a single text string per instance;
		// stitch multiple text parts together.
		var sb strings.Builder
		for _, part := range content {
			if part.Type != schemas.EmbeddingContentPartTypeText || part.Text == nil {
				return nil
			}
			sb.WriteString(*part.Text)
		}
		instance := VertexEmbeddingInstance{Content: sb.String()}
		if bifrostReq.Params != nil {
			instance.TaskType = bifrostReq.Params.TaskType
			instance.Title = bifrostReq.Params.Title
		}
		instances = append(instances, instance)
	}
	vertexReq.Instances = instances

	if bifrostReq.Params != nil {
		parameters := &VertexEmbeddingParameters{}
		autoTruncate := true
		if bifrostReq.Params.AutoTruncate != nil {
			autoTruncate = *bifrostReq.Params.AutoTruncate
		}
		parameters.AutoTruncate = &autoTruncate
		parameters.OutputDimensionality = bifrostReq.Params.Dimensions
		vertexReq.Parameters = parameters
	}

	return vertexReq
}

// ToVertexGeminiEmbeddingRequest converts a Bifrost embedding request to Vertex Gemini embedding format.
// Only a single content entry is supported (len == 1); batch is not supported by this endpoint.
func ToVertexGeminiEmbeddingRequest(bifrostReq *schemas.BifrostEmbeddingRequest) (*VertexGeminiEmbeddingRequest, error) {
	if bifrostReq == nil || bifrostReq.Input == nil || len(bifrostReq.Input.Contents) == 0 {
		return nil, fmt.Errorf("embedding input is not provided")
	}
	if len(bifrostReq.Input.Contents) > 1 {
		return nil, fmt.Errorf("vertex gemini embedding does not support batch inputs (multiple contents); use a single content entry")
	}

	content := bifrostReq.Input.Contents[0]
	params := bifrostReq.Params
	gemContent, err := gemini.EmbeddingContentToGeminiContent(content)
	if err != nil {
		return nil, err
	}
	req := &VertexGeminiEmbeddingRequest{
		Content: gemContent,
	}
	if params != nil {
		req.TaskType = params.TaskType
		req.Title = params.Title
		req.OutputDimensionality = params.Dimensions
		req.AutoTruncate = params.AutoTruncate

		if params.ExtraParams != nil {
			req.ExtraParams = params.ExtraParams
			if documentOCR, ok := schemas.SafeExtractBoolPointer(params.ExtraParams["documentOcr"]); ok {
				delete(req.ExtraParams, "documentOcr")
				req.DocumentOCR = documentOCR
			}
			if audioTrackExtraction, ok := schemas.SafeExtractBoolPointer(params.ExtraParams["audioTrackExtraction"]); ok {
				delete(req.ExtraParams, "audioTrackExtraction")
				req.AudioTrackExtraction = audioTrackExtraction
			}
		}
	}
	return req, nil
}

// extractBase64FromDataURI strips the "data:<mime>;base64," prefix from a data URI,
// returning the raw base64 string that Vertex multimodal embedding expects.
func extractBase64FromDataURI(dataURI string) string {
	if !strings.HasPrefix(dataURI, "data:") {
		return dataURI // already raw base64 or a GCS URI
	}
	info := schemas.ExtractURLTypeInfo(dataURI)
	if info.DataURLWithoutPrefix != nil {
		return *info.DataURLWithoutPrefix
	}
	return dataURI
}

// ToVertexMultimodalEmbeddingRequest converts a Bifrost embedding request to the
// Vertex native multimodal embedding format (multimodalembedding@001).
// Each content entry maps to one instance. Parts within a content are merged into
// the instance fields (text, image, video). Only text, image, and video are supported;
// audio and file parts will return an error.
func ToVertexMultimodalEmbeddingRequest(bifrostReq *schemas.BifrostEmbeddingRequest) (*VertexEmbeddingRequest, error) {
	if bifrostReq == nil || bifrostReq.Input == nil || len(bifrostReq.Input.Contents) == 0 {
		return nil, fmt.Errorf("embedding input is not provided")
	}

	instances := make([]VertexEmbeddingInstance, 0, len(bifrostReq.Input.Contents))
	for _, content := range bifrostReq.Input.Contents {
		instance := VertexEmbeddingInstance{}
		for _, part := range content {
			switch part.Type {
			case schemas.EmbeddingContentPartTypeText:
				instance.Text = part.Text
			case schemas.EmbeddingContentPartTypeImage:
				if part.Image == nil {
					return nil, fmt.Errorf("image part has no payload")
				}
				img := &VertexMultimodalImageInput{}
				if part.Image.Data != nil {
					b64 := extractBase64FromDataURI(*part.Image.Data)
					img.BytesBase64Encoded = &b64
				} else if part.Image.URL != nil {
					img.GCSUri = part.Image.URL
				} else {
					return nil, fmt.Errorf("image part must set data or url")
				}
				instance.Image = img
			case schemas.EmbeddingContentPartTypeVideo:
				if part.Video == nil {
					return nil, fmt.Errorf("video part has no payload")
				}
				if part.Video.URL == nil {
					return nil, fmt.Errorf("vertex multimodal embedding requires a GCS URI for video input")
				}
				vid := &VertexMultimodalVideoInput{GCSUri: part.Video.URL}
				instance.Video = vid
			default:
				return nil, fmt.Errorf("vertex multimodalembedding@001 does not support %q parts", part.Type)
			}
		}
		instances = append(instances, instance)
	}

	req := &VertexEmbeddingRequest{Instances: instances}
	if bifrostReq.Params != nil {
		req.Parameters = &VertexEmbeddingParameters{
			Dimension:    bifrostReq.Params.Dimensions,
			AutoTruncate: bifrostReq.Params.AutoTruncate,
		}
		req.ExtraParams = bifrostReq.Params.ExtraParams
	}
	return req, nil
}

// ToBifrostEmbeddingResponse converts a Vertex AI embedding response to Bifrost format.
// Handles both text embedding responses (Embeddings.Values) and native multimodal
// responses (TextEmbedding / ImageEmbedding / VideoEmbeddings).
func (response *VertexEmbeddingResponse) ToBifrostEmbeddingResponse() *schemas.BifrostEmbeddingResponse {
	if response == nil || len(response.Predictions) == 0 {
		return nil
	}

	embeddings := make([]schemas.EmbeddingData, 0, len(response.Predictions))
	var usage *schemas.BifrostLLMUsage
	idx := 0

	for _, prediction := range response.Predictions {
		// Text embedding model response
		if prediction.Embeddings != nil && len(prediction.Embeddings.Values) > 0 {
			embeddings = append(embeddings, schemas.EmbeddingData{
				Object:    "embedding",
				Embedding: schemas.EmbeddingsByType{Float: append([]float64(nil), prediction.Embeddings.Values...)},
				Index:     idx,
			})
			idx++
			if prediction.Embeddings.Statistics != nil {
				if usage == nil {
					usage = &schemas.BifrostLLMUsage{}
				}
				usage.TotalTokens += prediction.Embeddings.Statistics.TokenCount
				usage.PromptTokens += prediction.Embeddings.Statistics.TokenCount
			}
			continue
		}

		// Native multimodal model response — textEmbedding, imageEmbedding, videoEmbeddings
		// are all in the same embedding space so each is returned as a separate EmbeddingData.
		if len(prediction.TextEmbedding) > 0 {
			embeddings = append(embeddings, schemas.EmbeddingData{
				Object:    "embedding",
				Embedding: schemas.EmbeddingsByType{Float: append([]float64(nil), prediction.TextEmbedding...)},
				Index:     idx,
			})
			idx++
		}
		if len(prediction.ImageEmbedding) > 0 {
			embeddings = append(embeddings, schemas.EmbeddingData{
				Object:    "embedding",
				Embedding: schemas.EmbeddingsByType{Float: append([]float64(nil), prediction.ImageEmbedding...)},
				Index:     idx,
			})
			idx++
		}
		for _, ve := range prediction.VideoEmbeddings {
			embeddings = append(embeddings, schemas.EmbeddingData{
				Object:    "embedding",
				Embedding: schemas.EmbeddingsByType{Float: append([]float64(nil), ve.Embedding...)},
				Index:     idx,
			})
			idx++
		}
	}

	return &schemas.BifrostEmbeddingResponse{
		Object:      "list",
		Data:        embeddings,
		Usage:       usage,
		ExtraFields: schemas.BifrostResponseExtraFields{},
	}
}
