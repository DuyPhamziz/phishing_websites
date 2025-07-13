# phishing_websites
Đồ án Máy học ứng dụng CT294
Đề tài: Phát hiện Website Lừa đảo
Nguồn dataset: Phishing Websites của UCI
DOI: https://doi.org/10.24432/C51W2X 

LƯU Ý: 
Đây là đồ án xây dựng cho học phần trên lớp KHÔNG PHẢI SẢN PHẨN CHUYÊN NGHIỆP.
Là sản phẩm của nhóm sinh viên chuyên ngành công nghệ thông tin Khóa 49 của Đại học Cần Thơ (CTU).

THƯ MỤC DATA
Ở đây lưu dataset. Và chỉ sử dụng file phishing.arff để xử lí. (Vì file .old.arff đã được tác giả xử lí sẵn).

THƯ MỤC MODELS
Ở đây chứa các file code python của các thành viên (Mỗi thành viên 1 vài thuật toán). Đến cuối cùng, tổng hợp để có kết quả đối chiếu và đưa ra lựa chọn cuối cùng. (ALL_model_run.py và train_model.py)

Từ các kết quả của từng giải thuật, nhóm quyết định chọn Random Forest làm giải thuật chính. Và nó được xử lí ở các file ss.py (so sánh), kfold.py, code.py, input.py, outlier.py (xử lí dữ liệu trước và sau khi tiền xử lí. Các kết quả để đưa vào báo cáo)