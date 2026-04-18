"""
神经网络模型接口测试

验证 EnergyGradientNN 与 GPR 具有完全一致的接口
"""
import numpy as np
import sys

# 测试配置
test_config = {
    'hybrid': {
        'n_init': 5,
    },
    'gpr': {
        'local_radius': 0.5,
        'xi': 0.01,
        'lambda_grad': 0.1,
        'max_training_points': 30
    },
    'neural_network': {
        'hidden_layers': [32, 16],
        'activation': 'relu',
        'use_batchnorm': True,
        'dropout_rate': 0.1,
        'learning_rate': 0.001,
        'batch_size': 8,
        'max_epochs': 100,
        'early_stopping_patience': 20,
        'validation_split': 0.2,
        'energy_weight': 1.0,
        'gradient_weight': 0.1,
        'optimizer': 'adam',
        'weight_decay': 0.01,
        'normalize_input': True,
        'normalize_output': True
    },
    'optimizer': {
        'verbose': True
    }
}

def test_nn_interface():
    """测试神经网络接口"""
    print("=" * 60)
    print("测试 EnergyGradientNN 接口")
    print("=" * 60)
    
    try:
        from models.energy_gradient_nn import EnergyGradientNN
        from models.energy_gradient_gpr import EnergyGradientGPR
    except ImportError as e:
        print(f"❌ 导入失败：{e}")
        print("   请确保已安装 torch: pip install torch")
        return False
    
    # 设置维度（例如水分子：3 原子 * 3 = 9 维）
    dim = 9
    n_samples = 10
    
    # 创建模型
    print(f"\n1. 创建模型 (dim={dim})...")
    nn_model = EnergyGradientNN(test_config, dim)
    gpr_model = EnergyGradientGPR(test_config, dim)
    print(f"   ✓ NN 模型：{nn_model}")
    print(f"   ✓ GPR 模型：{gpr_model}")
    
    # 生成测试数据
    print(f"\n2. 生成测试数据 (n_samples={n_samples})...")
    np.random.seed(42)
    X = np.random.randn(n_samples, dim) * 0.1
    y = np.random.randn(n_samples) * 0.5
    gradients = np.random.randn(n_samples, dim) * 0.2
    
    print(f"   X 形状：{X.shape}")
    print(f"   y 形状：{y.shape}")
    print(f"   gradients 形状：{gradients.shape}")
    
    # 添加数据
    print(f"\n3. 添加训练数据...")
    for i in range(n_samples):
        nn_model.add_data(X[i], y[i], gradients[i])
        gpr_model.add_data(X[i], y[i], gradients[i])
    print(f"   ✓ NN 训练点数：{nn_model.n_training_points()}")
    print(f"   ✓ GPR 训练点数：{gpr_model.n_training_points()}")
    
    # 训练模型
    print(f"\n4. 训练模型...")
    print("   训练 NN 模型...", end=" ")
    nn_model.train(X, y, gradients)
    print(f"✓ (trained={nn_model.is_trained})")
    
    print("   训练 GPR 模型...", end=" ")
    gpr_model.train(X, y, gradients)
    print(f"✓ (trained={gpr_model.is_trained})")
    
    # 预测测试
    print(f"\n5. 预测测试...")
    x_test = np.random.randn(dim) * 0.1
    
    # predict_gradient 接口（核心接口）
    print("   测试 predict_gradient()...")
    nn_grad = nn_model.predict_gradient(x_test)
    gpr_grad = gpr_model.predict_gradient(x_test)
    print(f"   ✓ NN 梯度形状：{nn_grad.shape}, 范数：{np.linalg.norm(nn_grad):.6f}")
    print(f"   ✓ GPR 梯度形状：{gpr_grad.shape}, 范数：{np.linalg.norm(gpr_grad):.6f}")
    assert nn_grad.shape == (dim,), f"NN 梯度形状错误：{nn_grad.shape}"
    assert gpr_grad.shape == (dim,), f"GPR 梯度形状错误：{gpr_grad.shape}"
    
    # predict_energy_gradient 接口
    print("   测试 predict_energy_gradient()...")
    nn_energy, nn_grad2 = nn_model.predict_energy_gradient(x_test)
    gpr_energy, gpr_grad2 = gpr_model.predict_energy_gradient(x_test)
    print(f"   ✓ NN 能量：{nn_energy:.6f}, 梯度形状：{nn_grad2.shape}")
    print(f"   ✓ GPR 能量：{gpr_energy:.6f}, 梯度形状：{gpr_grad2.shape}")
    assert isinstance(nn_energy, float), f"NN 能量应为 float"
    assert nn_grad2.shape == (dim,), f"NN 梯度形状错误"
    
    # predict 接口
    print("   测试 predict()...")
    nn_energy2, nn_var = nn_model.predict(x_test)
    gpr_energy2, gpr_var = gpr_model.predict(x_test)
    print(f"   ✓ NN 能量：{nn_energy2:.6f}, 方差：{nn_var:.6f}")
    print(f"   ✓ GPR 能量：{gpr_energy2:.6f}, 方差：{gpr_var:.6f}")
    assert isinstance(nn_energy2, float), f"NN 能量应为 float"
    assert isinstance(nn_var, float), f"NN 方差应为 float"
    
    # acquisition_function 接口
    print("   测试 acquisition_function()...")
    nn_ei = nn_model.acquisition_function(x_test, y_min=-1.0)
    gpr_ei = gpr_model.acquisition_function(x_test, y_min=-1.0)
    print(f"   ✓ NN EI 值：{nn_ei:.6f}")
    print(f"   ✓ GPR EI 值：{gpr_ei:.6f}")
    
    # clear_data 接口
    print(f"\n6. 测试 clear_data()...")
    nn_model.clear_data()
    print(f"   ✓ NN 训练点数：{nn_model.n_training_points()}")
    assert nn_model.n_training_points() == 0, "clear_data 失败"
    assert nn_model.is_trained == False, "clear_data 后 is_trained 应为 False"
    
    print("\n" + "=" * 60)
    print("✅ 所有接口测试通过！")
    print("=" * 60)
    print("\n神经网络模型与 GPR 模型接口完全一致：")
    print("  - add_data(x, energy, gradient)")
    print("  - train(X, y, gradients)")
    print("  - predict(x) -> (energy, variance)")
    print("  - predict_gradient(x) -> gradient")
    print("  - predict_energy_gradient(x) -> (energy, gradient)")
    print("  - acquisition_function(x, y_min) -> ei")
    print("  - clear_data()")
    print("  - n_training_points()")
    return True


def test_data_flow():
    """测试数据流与 hybrid 优化器一致"""
    print("\n" + "=" * 60)
    print("测试数据流（模拟 hybrid 优化器）")
    print("=" * 60)
    
    from models.energy_gradient_nn import EnergyGradientNN
    
    dim = 9  # 3 原子分子
    nn_model = EnergyGradientNN(test_config, dim)
    
    # 模拟 hybrid 优化器的训练数据
    training_data = {
        'coords': [],
        'energy': [],
        'gradient': []
    }
    
    # 生成模拟数据（类似外层 L-BFGS 收集的数据）
    np.random.seed(42)
    for i in range(15):
        coord = np.random.randn(dim) * 0.1
        energy = np.random.randn() * 0.5
        gradient = np.random.randn(dim) * 0.2
        training_data['coords'].append(coord)
        training_data['energy'].append(energy)
        training_data['gradient'].append(gradient)
    
    print(f"\n1. 模拟训练数据：{len(training_data['coords'])} 个点")
    
    # 添加到模型
    for i in range(len(training_data['coords'])):
        nn_model.add_data(
            training_data['coords'][i],
            training_data['energy'][i],
            training_data['gradient'][i]
        )
    
    print(f"2. 添加到模型后：{nn_model.n_training_points()} 个点")
    
    # 训练
    X = np.array(training_data['coords'])
    y = np.array(training_data['energy'])
    gradients = np.array(training_data['gradient'])
    nn_model.train(X, y, gradients)
    
    # 模拟内层探索：使用预测梯度
    print(f"\n3. 模拟内层探索（5 步梯度下降）...")
    coords = np.random.randn(dim) * 0.1
    step_size = 0.05
    
    for step in range(5):
        gradient_pred = nn_model.predict_gradient(coords)
        g_norm = np.linalg.norm(gradient_pred)
        coords = coords - step_size * gradient_pred
        print(f"   Step {step+1}: |g|={g_norm:.6f}")
    
    print(f"   最终坐标形状：{coords.shape}")
    print(f"   ✓ 内层探索完成")
    
    print("\n" + "=" * 60)
    print("✅ 数据流测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = True
    success &= test_nn_interface()
    success &= test_data_flow()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！神经网络模型已就绪")
        print("=" * 60)
        print("\n使用方法:")
        print("  # 使用 GPR")
        print("  python main.py --method hybrid --molecule ethanol --ai_method gpr")
        print()
        print("  # 使用神经网络")
        print("  python main.py --method hybrid --molecule ethanol --ai_method nn")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ 测试失败")
        print("=" * 60)
        sys.exit(1)
