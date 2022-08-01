# Báo cáo notebook AI4Code
## Kết quả tốt nhất của nhóm
- Team: HCMUS - Atlantis
- Score: **0.8531**
![](https://i.imgur.com/Y7mxhY2.png)


Notebook của nhóm được tham khảo từ 2 tác giả
- **[Stronger baseline with code cells](https://www.kaggle.com/code/suicaokhoailang/stronger-baseline-with-code-cells)** của [suicaokhoailang](https://www.kaggle.com/suicaokhoailang)
- **[AI4Code Pairwise BertSmall inference](https://www.kaggle.com/code/yuanzhezhou/ai4code-pairwise-bertsmall-inference)** của [yuanzhezhou](https://www.kaggle.com/yuanzhezhou)

---

Dữ liệu các model mà nhóm đã chạy và có sử dụng trong notebook, thầy có thể [tải tại đây](Đã được đề cập ở báo cáo chính)

----
## Tóm tắt đề bài:

![](https://i.imgur.com/VIWReVH.png)


<!--  -->

## Tóm tắt quá trình thực hiện của nhóm

### Giai đoạn xây dựng mô hình

- Cả contest này đều dùng mô hình họ codebert, theo như khảo sát của nhóm thì toàn bộ các user phản hồi lại nhóm đều sử dụng một hoặc nhiều mô hình. Trong contest này cũng có sự tham gia của tác giả chính của codebert, đến từ lab của microsoft.

- Từ baseline sơ sài đầu tiên của anh Khôi (hay [suicaokhoailang]((https://www.kaggle.com/suicaokhoailang)) nhóm đã tiếp cận từ sớm và viết lại baseline này:
    - Thêm độ đo ktau cho mỗi epochs.
    - Thêm chế độ save epochs vì thời gian huấn luyện khá lâu.
    - Tùy chỉnh seed và một số tham số.
    - Thông qua quá trình EDA, nhóm nhận thấy tham số tốt hơn cho max-code-len là 23 thay vì 20 so với mặc định. Việc này giúp mô hình tăng 0.002~0.003 với cùng một số lượng epochs.


Toàn bộ mã nguồn train được cập nhật tại: https://github.com/t3bol90/ai4code-but-a-bunch-of-ensembles

- Nhóm sử dụng baseline của tác giả [The Devastator](https://www.kaggle.com/thedevastator) cho ý tưởng ensemble, vì thời gian giới hạn đối với một notebook submission (9 tiếng) nên nhóm sử dụng baseline này để tối ưu hóa việc sử dụng mô hình cho việc ensemble. Cụ thể, quá trình inference có thể diễn ra đối với một mô hình pointwise tầm 1.5 hrs và 4 hrs đối với mô hình pairwise. Từ đó nhóm có thể lựa chọn ensemble 3 mô hình pointwise và 1 mô hình pairwise hay  5 mô hình pointwise.

- Nhóm tiến hành huấn luyện 4 lớp mô hình, với số lượng và kỹ thuật tương ứng như sau:
    - Model 25 epochs codebert (được nhóm tiến hành train, nội dung code train được thể hiện trong file đính kèm)
    - Model 10 epochs codebert nhưng được train với tokenizer khác (MLM)
    - Model 10 epochs codebert (được nhóm sử dụng từ model public của một thành viên tham gia challenge)
    - Model public pairwaise

Nhóm nhận thấy rằng việc sử dụng một tokenizer khác (MLM) thay vì sử dụng default là mix giữa MLM+RTD cho ra kết quả hàm loss giảm tốt hơn và mô hình tăng evaluation score nhanh hơn qua từng epochs.

### Giai đoạn thử nghiệm

Để ensemble các kết quả từ các mô hình khác nhau, ta có thể chia làm hai nhóm tương ứng với bài toán này.
- Các mô hình pointwise: có thể ensemble trực tiếp kết quả score từ các mô hình dự đoán này và không làm thay đổi thứ tự của các code cell trước đó.
- Các mô hình pairwise: lấy trung bình vị trí của submission rank, việc này có thể dẫn tới thay đổi các vị trí code cell trước đó.


Để khắc phục vấn đề xáo trộn khi ensemble hai họ mô hình khác nhau, nhóm đã cài đặt một phiên bản ensemble không xáo trộn nhưng dẫn tới chi phí tính toán lớn, do đó không đủ thời gian để inference. Một thử nghiệm khác là nhóm ensemble các score predict của pointwise models trước, sau đó mới ensemble với pairwise, nhưng cách này vẫn không tốt bằng việc ensemble tất cả bằng việc lấy trung bình submission rank.

> Điều này dẫn tới đánh giá của nhóm về độ đo của challenge này sẽ đánh mạnh vào precision hơn là vào recall. Do đó việc khai thác từng order đúng sẽ không quan trọng bằng việc giảm tối thiểu tổng inversion.


Kết thúc giai đoạn thử nghiệm, nhóm có được baseline cơ bản bao gồm 4 mô hình và phương thức ensemble bằng cách lấy trung bình submission rank của từng model.

### Giai đoạn spam để tìm tỉ lệ:

Kết quả spam của các tỉ lệ được đề cập ở: [AI4Code Sheet](Đã được đề cập ở báo cáo chính)


<!--  -->


## Giải thích: Hướng tiếp cận pairwise
![](https://i.imgur.com/VTXzheh.png)


## Giải thích: Hướng tiếp cận pointwise
![](https://i.imgur.com/NdKhK93.png)



## Acknowledgement

Quá quá trình làm việc, nhóm đã trải qua nhiều lần bế tắc trong việc tiếp tục train codebert hay tìm một cách khác để tăng được kết quả. Nhóm đã thu nhận được rất nhiều insight (nhờ mạnh dạn đi spam hỏi toàn bộ những kagglers hơn mình thông qua email và Linkedin) từ các Kaggler khác, trong đó xin chân thành cảm ơn các phản hồi giá trị:

- [Xianchao Wu](https://www.kaggle.com/xianchaowu), một PhD và một kỹ sư AI cho NVIDIA Nhật Bản. Nhóm đã có được insight có thể thực hiện multi-gpu với torch-distributed. Một kỹ thuật đòi hỏi điều kiện phần cứng cao nên đã không thực hiện được trong đồ án này, nhưng việc học hỏi được một kỹ thuật tăng tốc như vậy sẽ giúp nhóm trong chặng đường nghiên cứu lâu dài phía trước.

- [Sergio Manuel Papadakis](https://www.kaggle.com/socom20), một kỹ sư AI và Hạt Nhân ở Argentina. Nhóm đã có được insight của phương pháp ensemble và tối ưu thời gian ensemble cho từng mô hình thông qua disscuss với kaggler này.

