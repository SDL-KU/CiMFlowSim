#include "cnn_strategy.h"

// Factory function for tiling strategy
CNNStrategy* create_tiling_strategy(const TilingConfig& tiling) {
    return new TilingStrategy(tiling);
}
