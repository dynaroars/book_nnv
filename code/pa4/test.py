from objective import parse_vnnlib

if __name__ == "__main__":
    input_shape = (1, 5)
    properties = parse_vnnlib("prop_7.vnnlib", input_shape)
    
    sub_property = properties.pop(1)
    print(sub_property.lower_bounds)
    # tensor([[-0.3284, -0.5000, -0.5000, -0.5000, -0.5000]])
    
    print(sub_property.upper_bounds)
    # tensor([[0.6799, 0.5000, 0.5000, 0.5000, 0.5000]])

    print(sub_property.cs)
    # tensor([[[-1.,  0.,  0.,  1.,  0.],
    #         [ 0., -1.,  0.,  1.,  0.],
    #         [ 0.,  0., -1.,  1.,  0.]]])
    
    print(sub_property.rhs)
    # tensor([[0., 0., 0.]])