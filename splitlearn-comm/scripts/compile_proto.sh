#!/bin/bash
# 编译 Protocol Buffer 文件

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

log_info "Compiling Protocol Buffer files..."

# 检查 grpcio-tools 是否安装
if ! python3 -c "import grpc_tools" 2>/dev/null; then
    log_warn "grpcio-tools not found, installing..."
    pip3 install grpcio-tools
fi

# 编译 protobuf
PROTO_DIR="src/splitlearn_comm/protocol/protos"
OUTPUT_DIR="src/splitlearn_comm/protocol"

python3 -m grpc_tools.protoc \
    --proto_path="$PROTO_DIR" \
    --python_out="$OUTPUT_DIR" \
    --grpc_python_out="$OUTPUT_DIR" \
    --pyi_out="$OUTPUT_DIR" \
    "$PROTO_DIR/compute_service.proto"

# 修复导入路径（如果需要）
if [ -f "$OUTPUT_DIR/compute_service_pb2_grpc.py" ]; then
    # 修复相对导入
    sed -i.bak 's/import compute_service_pb2/from . import compute_service_pb2/' "$OUTPUT_DIR/compute_service_pb2_grpc.py" 2>/dev/null || \
    sed -i '' 's/import compute_service_pb2/from . import compute_service_pb2/' "$OUTPUT_DIR/compute_service_pb2_grpc.py"
    rm -f "$OUTPUT_DIR/compute_service_pb2_grpc.py.bak"
fi

log_info "✓ Protocol Buffer compilation completed"
log_info "Generated files:"
log_info "  - compute_service_pb2.py"
log_info "  - compute_service_pb2_grpc.py"
log_info "  - compute_service_pb2.pyi"
