#include "kernel_operator.h"
#include <type_traits>           
using namespace AscendC;      
constexpr int32_t BUFFER_NUM = 2; 
template<typename T> struct Map {using type = T;};      
template<> struct Map<int8_t> {using type = half;};                    
template<typename TYPE_Y, typename TYPE_DY, typename TYPE_Z> class KernelAsinhGrad_Defined { 
    using T = TYPE_Z;          
public:                   
    __aicore__ inline KernelAsinhGrad_Defined() {}
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR dy, GM_ADDR z,
                                uint32_t ALIGN_NUM, uint32_t block_size, uint32_t core_size, uint32_t core_remain) {            
                                          
        this->blockLength = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);  
        this->tileLength = block_size;                                                          
        this->blockLength = this->blockLength + (this->blockLength % ALIGN_NUM ? ALIGN_NUM - this->blockLength % ALIGN_NUM : 0);              
        auto startPointer = core_size * GetBlockIdx();  
        auto bufferlength = this->blockLength;
        Gm_y.SetGlobalBuffer((__gm__ TYPE_Y*)y + startPointer, bufferlength);
        Gm_dy.SetGlobalBuffer((__gm__ TYPE_DY*)dy + startPointer, bufferlength); 
        Gm_z.SetGlobalBuffer((__gm__ TYPE_Z*)z + startPointer, bufferlength); 
        this->tileNum = this->blockLength / this->tileLength + (this->blockLength % this->tileLength > 0);
        pipe.InitBuffer(Q_y, BUFFER_NUM, this->tileLength * sizeof(TYPE_Y));
        pipe.InitBuffer(Q_dy, BUFFER_NUM, this->tileLength * sizeof(TYPE_DY)); 
        pipe.InitBuffer(Q_z, BUFFER_NUM, this->tileLength * sizeof(TYPE_Z)); 
        pipe.InitBuffer(tmpYBuffer, this->tileLength * sizeof(float)); 
        pipe.InitBuffer(tmpDYBuffer, this->tileLength * sizeof(float)); 
        pipe.InitBuffer(tmpZBuffer, this->tileLength * sizeof(float));
    }    
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount-1; i++) {   
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        uint32_t length = this->blockLength - this->tileLength * (loopCount - 1);
        CopyIn(loopCount - 1, length);  
        Compute(loopCount - 1, length); 
        CopyOut(loopCount - 1, length); 
    }
        
private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y>  y = Q_y.AllocTensor<TYPE_Y>(); 
        LocalTensor<TYPE_DY>  dy = Q_dy.AllocTensor<TYPE_DY>(); 
        DataCopy(y, Gm_y[progress * this->tileLength], length); 
        DataCopy(dy, Gm_dy[progress * this->tileLength], length);
        Q_y.EnQue(y);
        Q_dy.EnQue(dy);
    }
   __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Y> y = Q_y.DeQue<TYPE_Y>();
        LocalTensor<TYPE_DY> dy = Q_dy.DeQue<TYPE_DY>();
        LocalTensor<TYPE_Z> z = Q_z.AllocTensor<TYPE_Z>();   
        LocalTensor<float> tmpY  = tmpYBuffer.Get<float>();
        LocalTensor<float> tmpDY  = tmpDYBuffer.Get<float>();
        LocalTensor<float> tmpZ  = tmpZBuffer.Get<float>();  
        
        if constexpr (std::is_same_v<T, half>){ 
            Cast(tmpY, y, RoundMode::CAST_NONE, length);
            Cast(tmpDY, dy, RoundMode::CAST_NONE, length); 
            Cast(tmpZ, z, RoundMode::CAST_NONE, length); 
            Exp(tmpZ, tmpY, length); 
            Muls(tmpY, tmpY, static_cast<float>(-1), length);
            Exp(tmpY, tmpY, length);
            Add(tmpY, tmpZ, tmpY, length);
            Muls(tmpDY, tmpDY, static_cast<float>(2), length);
            Div(tmpZ, tmpDY, tmpY, length);
            Cast(z, tmpZ, RoundMode::CAST_CEIL, length); 
        }else if constexpr (std::is_same_v<T, float>){ 
            Exp(z, y, length); 
            Muls(y, y, static_cast<TYPE_Z>(-1), length);
            Exp(y, y, length);
            Add(y, z, y, length);
            Muls(dy, dy, static_cast<TYPE_Z>(2), length);
            Div(z, dy, y, length);
        }
        
        Q_y.FreeTensor(y);
        Q_dy.FreeTensor(dy);
        Q_z.EnQue<TYPE_Z>(z); 
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<TYPE_Z> z = Q_z.DeQue<TYPE_Z>();         
        DataCopy(Gm_z[progress * this->tileLength], z, length);
        Q_z.FreeTensor(z);
    } 

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM>  Q_y, Q_dy;  
    TQue<QuePosition::VECOUT, BUFFER_NUM> Q_z;
    TBuf<QuePosition::VECCALC> tmpYBuffer,tmpDYBuffer, tmpZBuffer;   

    GlobalTensor<TYPE_Y> Gm_y;
    GlobalTensor<TYPE_DY> Gm_dy;
    GlobalTensor<TYPE_Z> Gm_z; 

    uint32_t blockLength;  
    uint32_t tileNum;
    uint32_t tileLength;    

}; 

extern "C" __global__ __aicore__ void asinh_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);  

     KernelAsinhGrad_Defined<DTYPE_Y, DTYPE_DY, DTYPE_Z> op;  
    op.Init(y, dy, z,  
            tiling_data.ALIGN_NUM, tiling_data.block_size, tiling_data.core_size, tiling_data.core_remain);   
    op.Process();      
}
 

 