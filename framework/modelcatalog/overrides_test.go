package modelcatalog

import (
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	configstoreTables "github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type noOpLogger struct{}

func (noOpLogger) Debug(string, ...any)                   {}
func (noOpLogger) Info(string, ...any)                    {}
func (noOpLogger) Warn(string, ...any)                    {}
func (noOpLogger) Error(string, ...any)                   {}
func (noOpLogger) Fatal(string, ...any)                   {}
func (noOpLogger) SetLevel(schemas.LogLevel)              {}
func (noOpLogger) SetOutputType(schemas.LoggerOutputType) {}
func (noOpLogger) LogHTTPRequest(schemas.LogLevel, string) schemas.LogEventBuilder {
	return schemas.NoopLogEvent
}

func TestSetProviderPricingOverrides_InvalidRegex(t *testing.T) {
	t.Skip()
	mc := newTestCatalog(nil, nil)
	err := mc.SetProviderPricingOverrides(schemas.OpenAI, []schemas.ProviderPricingOverride{
		{
			ModelPattern: "[",
			MatchType:    schemas.PricingOverrideMatchRegex,
		},
	})
	require.Error(t, err)
}

func TestGetPricing_OverridePrecedenceExactWildcardRegex(t *testing.T) {
	t.Skip()
	mc := newTestCatalog(nil, nil)
	mc.logger = noOpLogger{}
	mc.pricingData[makeKey("gpt-4o", "openai", "chat")] = configstoreTables.TableModelPricing{
		Model:              "gpt-4o",
		Provider:           "openai",
		Mode:               "chat",
		InputCostPerToken:  1,
		OutputCostPerToken: 2,
	}

	exact := 20.0
	wildcard := 10.0
	regex := 30.0
	require.NoError(t, mc.SetProviderPricingOverrides(schemas.OpenAI, []schemas.ProviderPricingOverride{
		{
			ModelPattern:      "gpt-*",
			MatchType:         schemas.PricingOverrideMatchWildcard,
			InputCostPerToken: &wildcard,
		},
		{
			ModelPattern:      "^gpt-.*$",
			MatchType:         schemas.PricingOverrideMatchRegex,
			InputCostPerToken: &regex,
		},
		{
			ModelPattern:      "gpt-4o",
			MatchType:         schemas.PricingOverrideMatchExact,
			InputCostPerToken: &exact,
		},
	}))

	pricing, ok := mc.getPricing("gpt-4o", "openai", schemas.ChatCompletionRequest)
	require.True(t, ok)
	require.NotNil(t, pricing)
	assert.Equal(t, 20.0, pricing.InputCostPerToken)
	assert.Equal(t, 2.0, pricing.OutputCostPerToken)
}

func TestGetPricing_WildcardBeatsRegex(t *testing.T) {
	t.Skip()
	mc := newTestCatalog(nil, nil)
	mc.logger = noOpLogger{}
	mc.pricingData[makeKey("gpt-4o-mini", "openai", "chat")] = configstoreTables.TableModelPricing{
		Model:              "gpt-4o-mini",
		Provider:           "openai",
		Mode:               "chat",
		InputCostPerToken:  1,
		OutputCostPerToken: 2,
	}

	wildcard := 11.0
	regex := 12.0
	require.NoError(t, mc.SetProviderPricingOverrides(schemas.OpenAI, []schemas.ProviderPricingOverride{
		{
			ModelPattern:      "^gpt-4o.*$",
			MatchType:         schemas.PricingOverrideMatchRegex,
			InputCostPerToken: &regex,
		},
		{
			ModelPattern:      "gpt-4o*",
			MatchType:         schemas.PricingOverrideMatchWildcard,
			InputCostPerToken: &wildcard,
		},
	}))

	pricing, ok := mc.getPricing("gpt-4o-mini", "openai", schemas.ChatCompletionRequest)
	require.True(t, ok)
	require.NotNil(t, pricing)
	assert.Equal(t, 11.0, pricing.InputCostPerToken)
}

func TestGetPricing_RequestTypeSpecificOverrideBeatsGeneric(t *testing.T) {
	t.Skip()
	mc := newTestCatalog(nil, nil)
	mc.logger = noOpLogger{}
	mc.pricingData[makeKey("gpt-4o", "openai", "responses")] = configstoreTables.TableModelPricing{
		Model:              "gpt-4o",
		Provider:           "openai",
		Mode:               "responses",
		InputCostPerToken:  1,
		OutputCostPerToken: 2,
	}

	specific := 15.0
	generic := 9.0
	require.NoError(t, mc.SetProviderPricingOverrides(schemas.OpenAI, []schemas.ProviderPricingOverride{
		{
			ModelPattern:      "gpt-4o",
			MatchType:         schemas.PricingOverrideMatchExact,
			InputCostPerToken: &generic,
		},
		{
			ModelPattern:      "gpt-4o",
			MatchType:         schemas.PricingOverrideMatchExact,
			RequestTypes:      []schemas.RequestType{schemas.ResponsesRequest},
			InputCostPerToken: &specific,
		},
	}))

	pricing, ok := mc.getPricing("gpt-4o", "openai", schemas.ResponsesRequest)
	require.True(t, ok)
	require.NotNil(t, pricing)
	assert.Equal(t, 15.0, pricing.InputCostPerToken)
}

func TestGetPricing_AppliesOverrideAfterFallbackResolution(t *testing.T) {
	t.Skip()
	mc := newTestCatalog(nil, nil)
	mc.logger = noOpLogger{}
	mc.pricingData[makeKey("gpt-4o", "vertex", "chat")] = configstoreTables.TableModelPricing{
		Model:              "gpt-4o",
		Provider:           "vertex",
		Mode:               "chat",
		InputCostPerToken:  1,
		OutputCostPerToken: 2,
	}

	override := 7.0
	require.NoError(t, mc.SetProviderPricingOverrides(schemas.Gemini, []schemas.ProviderPricingOverride{
		{
			ModelPattern:      "gpt-4o",
			MatchType:         schemas.PricingOverrideMatchExact,
			InputCostPerToken: &override,
		},
	}))

	pricing, ok := mc.getPricing("gpt-4o", "gemini", schemas.ChatCompletionRequest)
	require.True(t, ok)
	require.NotNil(t, pricing)
	assert.Equal(t, 7.0, pricing.InputCostPerToken)
}

func TestGetPricing_ExactOverrideDoesNotMatchProviderPrefixedModel(t *testing.T) {
	t.Skip()
	mc := newTestCatalog(nil, nil)
	mc.logger = noOpLogger{}
	mc.pricingData[makeKey("openai/gpt-4o", "openai", "chat")] = configstoreTables.TableModelPricing{
		Model:              "openai/gpt-4o",
		Provider:           "openai",
		Mode:               "chat",
		InputCostPerToken:  1,
		OutputCostPerToken: 2,
	}

	override := 19.0
	require.NoError(t, mc.SetProviderPricingOverrides(schemas.OpenAI, []schemas.ProviderPricingOverride{
		{
			ModelPattern:      "gpt-4o",
			MatchType:         schemas.PricingOverrideMatchExact,
			InputCostPerToken: &override,
		},
	}))

	pricing, ok := mc.getPricing("openai/gpt-4o", "openai", schemas.ChatCompletionRequest)
	require.True(t, ok)
	require.NotNil(t, pricing)
	assert.Equal(t, 1.0, pricing.InputCostPerToken)
}

func TestGetPricing_NoMatchingOverrideLeavesPricingUnchanged(t *testing.T) {
	t.Skip()
	mc := newTestCatalog(nil, nil)
	mc.logger = noOpLogger{}
	baseCacheRead := 0.4
	mc.pricingData[makeKey("gpt-4o", "openai", "chat")] = configstoreTables.TableModelPricing{
		Model:                   "gpt-4o",
		Provider:                "openai",
		Mode:                    "chat",
		InputCostPerToken:       1,
		OutputCostPerToken:      2,
		CacheReadInputTokenCost: &baseCacheRead,
	}

	override := 9.0
	require.NoError(t, mc.SetProviderPricingOverrides(schemas.OpenAI, []schemas.ProviderPricingOverride{
		{
			ModelPattern:      "claude-*",
			MatchType:         schemas.PricingOverrideMatchWildcard,
			InputCostPerToken: &override,
		},
	}))

	pricing, ok := mc.getPricing("gpt-4o", "openai", schemas.ChatCompletionRequest)
	require.True(t, ok)
	require.NotNil(t, pricing)
	assert.Equal(t, 1.0, pricing.InputCostPerToken)
	assert.Equal(t, 2.0, pricing.OutputCostPerToken)
	require.NotNil(t, pricing.CacheReadInputTokenCost)
	assert.Equal(t, 0.4, *pricing.CacheReadInputTokenCost)
}

func TestDeleteProviderPricingOverrides_StopsApplying(t *testing.T) {
	t.Skip()
	mc := newTestCatalog(nil, nil)
	mc.logger = noOpLogger{}
	mc.pricingData[makeKey("gpt-4o", "openai", "chat")] = configstoreTables.TableModelPricing{
		Model:              "gpt-4o",
		Provider:           "openai",
		Mode:               "chat",
		InputCostPerToken:  1,
		OutputCostPerToken: 2,
	}

	override := 11.0
	require.NoError(t, mc.SetProviderPricingOverrides(schemas.OpenAI, []schemas.ProviderPricingOverride{
		{
			ModelPattern:      "gpt-4o",
			MatchType:         schemas.PricingOverrideMatchExact,
			InputCostPerToken: &override,
		},
	}))

	pricing, ok := mc.getPricing("gpt-4o", "openai", schemas.ChatCompletionRequest)
	require.True(t, ok)
	require.NotNil(t, pricing)
	assert.Equal(t, 11.0, pricing.InputCostPerToken)

	mc.DeleteProviderPricingOverrides(schemas.OpenAI)

	pricing, ok = mc.getPricing("gpt-4o", "openai", schemas.ChatCompletionRequest)
	require.True(t, ok)
	require.NotNil(t, pricing)
	assert.Equal(t, 1.0, pricing.InputCostPerToken)
}

func TestGetPricing_WildcardSpecificityLongerLiteralWins(t *testing.T) {
	t.Skip()
	mc := newTestCatalog(nil, nil)
	mc.logger = noOpLogger{}
	mc.pricingData[makeKey("gpt-4o-mini", "openai", "chat")] = configstoreTables.TableModelPricing{
		Model:              "gpt-4o-mini",
		Provider:           "openai",
		Mode:               "chat",
		InputCostPerToken:  1,
		OutputCostPerToken: 2,
	}

	generic := 5.0
	specific := 6.0
	require.NoError(t, mc.SetProviderPricingOverrides(schemas.OpenAI, []schemas.ProviderPricingOverride{
		{
			ModelPattern:      "gpt-*",
			MatchType:         schemas.PricingOverrideMatchWildcard,
			InputCostPerToken: &generic,
		},
		{
			ModelPattern:      "gpt-4o*",
			MatchType:         schemas.PricingOverrideMatchWildcard,
			InputCostPerToken: &specific,
		},
	}))

	pricing, ok := mc.getPricing("gpt-4o-mini", "openai", schemas.ChatCompletionRequest)
	require.True(t, ok)
	require.NotNil(t, pricing)
	assert.Equal(t, 6.0, pricing.InputCostPerToken)
}

func TestGetPricing_ConfigOrderTiebreakFirstWinsWhenEqual(t *testing.T) {
	t.Skip()
	mc := newTestCatalog(nil, nil)
	mc.logger = noOpLogger{}
	mc.pricingData[makeKey("gpt-4o-mini", "openai", "chat")] = configstoreTables.TableModelPricing{
		Model:              "gpt-4o-mini",
		Provider:           "openai",
		Mode:               "chat",
		InputCostPerToken:  1,
		OutputCostPerToken: 2,
	}

	first := 8.0
	second := 9.0
	require.NoError(t, mc.SetProviderPricingOverrides(schemas.OpenAI, []schemas.ProviderPricingOverride{
		{
			ModelPattern:      "gpt-4o*",
			MatchType:         schemas.PricingOverrideMatchWildcard,
			InputCostPerToken: &first,
		},
		{
			ModelPattern:      "gpt-4o*",
			MatchType:         schemas.PricingOverrideMatchWildcard,
			InputCostPerToken: &second,
		},
	}))

	pricing, ok := mc.getPricing("gpt-4o-mini", "openai", schemas.ChatCompletionRequest)
	require.True(t, ok)
	require.NotNil(t, pricing)
	assert.Equal(t, 8.0, pricing.InputCostPerToken)
}

func TestPatchPricing_PartialPatchOnlyChangesSpecifiedFields(t *testing.T) {
	t.Skip()
	baseCacheRead := 0.4
	baseInputImage := 0.7
	base := configstoreTables.TableModelPricing{
		Model:                   "gpt-4o",
		Provider:                "openai",
		Mode:                    "chat",
		InputCostPerToken:       1,
		OutputCostPerToken:      2,
		CacheReadInputTokenCost: &baseCacheRead,
		InputCostPerImage:       &baseInputImage,
	}

	patched := patchPricing(base, schemas.ProviderPricingOverride{
		ModelPattern:            "gpt-4o",
		MatchType:               schemas.PricingOverrideMatchExact,
		InputCostPerToken:       schemas.Ptr(3.0),
		CacheReadInputTokenCost: schemas.Ptr(0.9),
	})

	// Changed fields
	assert.Equal(t, 3.0, patched.InputCostPerToken)
	require.NotNil(t, patched.CacheReadInputTokenCost)
	assert.Equal(t, 0.9, *patched.CacheReadInputTokenCost)

	// Unchanged fields
	assert.Equal(t, 2.0, patched.OutputCostPerToken)
	require.NotNil(t, patched.InputCostPerImage)
	assert.Equal(t, 0.7, *patched.InputCostPerImage)
}

func TestPatchPricing_272kFields(t *testing.T) {
	base := configstoreTables.TableModelPricing{
		Model:              "claude-3-7-sonnet",
		Provider:           "anthropic",
		Mode:               "chat",
		InputCostPerToken:  0.000003,
		OutputCostPerToken: 0.000015,
	}

	patched := patchPricing(base, schemas.ProviderPricingOverride{
		ModelPattern:                               "claude-3-7-sonnet",
		MatchType:                                  schemas.PricingOverrideMatchExact,
		InputCostPerTokenAbove272kTokens:           schemas.Ptr(0.000009),
		InputCostPerTokenAbove272kTokensPriority:   schemas.Ptr(0.000012),
		OutputCostPerTokenAbove272kTokens:          schemas.Ptr(0.000045),
		OutputCostPerTokenAbove272kTokensPriority:  schemas.Ptr(0.00006),
		CacheReadInputTokenCostAbove272kTokens:     schemas.Ptr(0.0000009),
		CacheReadInputTokenCostAbove272kTokensPriority: schemas.Ptr(0.0000012),
	})

	// All 272k fields applied
	require.NotNil(t, patched.InputCostPerTokenAbove272kTokens)
	assert.Equal(t, 0.000009, *patched.InputCostPerTokenAbove272kTokens)
	require.NotNil(t, patched.InputCostPerTokenAbove272kTokensPriority)
	assert.Equal(t, 0.000012, *patched.InputCostPerTokenAbove272kTokensPriority)
	require.NotNil(t, patched.OutputCostPerTokenAbove272kTokens)
	assert.Equal(t, 0.000045, *patched.OutputCostPerTokenAbove272kTokens)
	require.NotNil(t, patched.OutputCostPerTokenAbove272kTokensPriority)
	assert.Equal(t, 0.00006, *patched.OutputCostPerTokenAbove272kTokensPriority)
	require.NotNil(t, patched.CacheReadInputTokenCostAbove272kTokens)
	assert.Equal(t, 0.0000009, *patched.CacheReadInputTokenCostAbove272kTokens)
	require.NotNil(t, patched.CacheReadInputTokenCostAbove272kTokensPriority)
	assert.Equal(t, 0.0000012, *patched.CacheReadInputTokenCostAbove272kTokensPriority)

	// Base rates unchanged
	assert.Equal(t, 0.000003, patched.InputCostPerToken)
	assert.Equal(t, 0.000015, patched.OutputCostPerToken)
}

func TestPatchPricing_272kFieldsNilDoNotOverwrite(t *testing.T) {
	// Existing 272k rates are not cleared when the override doesn't specify them
	base := configstoreTables.TableModelPricing{
		Model:                            "claude-3-7-sonnet",
		Provider:                         "anthropic",
		Mode:                             "chat",
		InputCostPerToken:                0.000003,
		OutputCostPerToken:               0.000015,
		InputCostPerTokenAbove272kTokens: schemas.Ptr(0.000009),
		OutputCostPerTokenAbove272kTokens: schemas.Ptr(0.000045),
	}

	patched := patchPricing(base, schemas.ProviderPricingOverride{
		ModelPattern:      "claude-3-7-sonnet",
		MatchType:         schemas.PricingOverrideMatchExact,
		InputCostPerToken: schemas.Ptr(0.000005),
	})

	// Base rate changed
	assert.Equal(t, 0.000005, patched.InputCostPerToken)

	// 272k rates preserved since override left them nil
	require.NotNil(t, patched.InputCostPerTokenAbove272kTokens)
	assert.Equal(t, 0.000009, *patched.InputCostPerTokenAbove272kTokens)
	require.NotNil(t, patched.OutputCostPerTokenAbove272kTokens)
	assert.Equal(t, 0.000045, *patched.OutputCostPerTokenAbove272kTokens)
}
