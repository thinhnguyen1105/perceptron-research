## PERCEPTRON ALGORITHM
    1. Perceptron là gì ?
    - nguồn gốc của Deep Learning (Neutral Network)
    - thuật toán Classification cho trường hợp chỉ có 2 class
    - thuật toán có input đầu vào là nhiều tín hiệu khác nhau và đầu ra là 1 tín hiệu duy nhất
    - tín hiệu của perceptron (0/1) - binary với 0 là không truyền tín hiệu 1 là có truyền tín hiệu

    2.Thành phần của perceptron (5 thành phần chính)
    - x1, x2 là tín hiệu input
    - y là tín hiệu đầu ra
    - weight là trọng số của mỗi tín hiệu
    - các node hay neuron
    - threadhold là ngưỡng

    3. Cách thức hoạt động
    if ((x1.w1 + x2.w2) <= threadhold ) {
        return 0
    } else {
        return 1
    }

    4. Ví dụ với mạch AND, NAND, OR và XOR
    - AND, NAND và OR là các mạch cơ bản
    - XOR là mạch chồng lên nhau, đây cũng chính là điểm mạnh của perceptron
    - Thực hành: Code ví dụ ( /perceptron-basic/example.py, multiple-stage.py )

    5. Xây dựng hàm mất mát
    - trường hợp có những điểm bị phân lớp lỗi ( missmisclassified )
    - solution ~> đếm số lượng điểm bị missclasified và tối thiểu hàm số này

    6. Công thức cụ thể
    - có yi = 1 nếu xi thuộc class xanh và yi = - 1 nếu xi thuộc class đỏ
    - với label(x) = 1 if wTx với x>=0 , otherwise -1 ( hàm xác định dấu )
    - vậy khi có điểm missclassified thì hàm -yi*sgn(wTx) là hàm đếm số lượng các điểm đó ~> ta phải đi tối thiểu hàm số này

###PROBLEM
    1. Problem
    - Khi có điểm bị missclasified, hàm số này sẽ = 1 vì yi và sgn() trái dấu với nhau, nhưng không tính được đạo hàm của hàm này theo w 
    - Xét hàm mất mát -yi*wTxi khi bỏ đi hàm sgn() thì mỗi điểm missclassified càng xa thì giá trị của hàm này càng lớn. Vậy đơn giản là ta cần đi tối thiểu hàm số này

    2. Solution

    Với mỗi điểm missclassified ta có:
    - yi*wT*xi ~> đạo hàm hàm này ta có: -yi*xi
    kết luận: Nếu xi bị missclassified thì cập nhật lại theo công thức
    w = w + yi*xi
    - Kiểm tra xem còn điểm nào bị missclassified nếu có quay lại bước 2

    
    


     