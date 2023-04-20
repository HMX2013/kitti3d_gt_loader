#include "tools/utils.hpp"
#include <boost/format.hpp>

#ifndef PATCHWORK_PCD_LOADER_HPP
#define PATCHWORK_PCD_LOADER_HPP


class KittiLoader {
public:
    KittiLoader(const std::string &bin_path, const std::string &label_path) {
        pc_path_ = bin_path;
        label_path_ = label_path;

        for (num_frames_ = 0;; num_frames_++) {
            std::string filename = (boost::format("%s/%06d.bin") % pc_path_ % num_frames_).str();
            if (!boost::filesystem::exists(filename)) {
                break;
            }
        }

        for (num_labels_ = 0;; num_labels_++) {
            std::string filename = (boost::format("%s/%06d.txt") % label_path_ % num_labels_).str();
            if (!boost::filesystem::exists(filename)) {
                break;
            }
        }

        if (num_frames_ == 0) {
            std::cerr << "\033[1;31mError: No files in " << pc_path_ << "\033[0m" << std::endl;
        }

        std::cout << "Total " << num_frames_ << " files are loaded" << std::endl;
    }

    ~KittiLoader() {}

    size_t size() const { return num_labels_; }

    template<typename T>
    void get_cloud(std::string &labeled_bin, pcl::PointCloud<T> &cloud) const {
        FILE *file = fopen(labeled_bin.c_str(), "rb");
        if (!file) {
            throw std::invalid_argument("Could not open the .bin file!");
        }
        std::vector<float> buffer(1000000);
        size_t num_points = fread(reinterpret_cast<char *>(buffer.data()), sizeof(float), buffer.size(), file) / 4;
        fclose(file);

        cloud.points.resize(num_points);
        if (std::is_same<T, pcl::PointXYZ>::value) {
            for (int i = 0; i < num_points; i++) {
                auto &pt = cloud.at(i);
                pt.x = buffer[i * 4];
                pt.y = buffer[i * 4 + 1];
                pt.z = buffer[i * 4 + 2];
            }
        }
    }

    void get_gt_label(size_t idx, size_t idx_seq, std::vector<std::string> &labeled_bin, std::vector<double> &theta_kitti)
    {
        std::string label_name = (boost::format("%s/%06d.txt") % label_path_ % idx).str();
        // std::cout <<"label_name is "<< label_name << std::endl;

        std::ifstream label_input; 
        label_input.open(label_name.data()); 

        std::string str_line;
        std::vector<std::string> label_string;

        while(getline(label_input,str_line))
        {
            label_string.push_back(str_line);
        }
        label_input.close();

        bool valid_label;
        for (size_t i = 0; i < label_string.size(); i++)
        {
            valid_label = true;

            std::istringstream divid_sp(label_string[i]);
            std::vector<std::string> labels_ith;
            std::string label;
            while(divid_sp >> label) {
                labels_ith.push_back(label);
            }

            // if (labels_ith[0] == "DontCare")
            //     valid_label = false;
            
            if (i != idx_seq)
                valid_label = false;

            if (idx_seq == -1)
                valid_label = true;

            if (labels_ith[0] == "Car" || labels_ith[0] == "Van" || labels_ith[0] == "Truck")
                valid_label = true;
            else
                valid_label = false;

            if (valid_label)
            {
                std::string valid_label_string = (boost::format("%s/%06d_%s_%d.bin") % pc_path_ % idx % labels_ith[0] % i).str();
                labeled_bin.push_back(valid_label_string);
                std::istringstream string2double(labels_ith[14]);
                double theta_double;
                string2double >> theta_double;
                theta_kitti.push_back(theta_double);
            }
        }
        // std::cout << "---------------------" << std::endl;
    }

    void debug(std::string &debug_path, std::vector<int> &debug_seq, std::vector<int> &debug_seq_2)
    {

        std::ifstream label_input; 
        label_input.open(debug_path); 

        std::string str_line;
        std::vector<std::string> label_string;

        while(getline(label_input,str_line))
        {
            label_string.push_back(str_line);
        }
        label_input.close();

        for (size_t i = 0; i < label_string.size(); i++)
        {
            std::cout << "label_string is " << label_string[i] << std::endl;
            double seq;
            std::istringstream divid_sp(label_string[i]);
            std::vector<double> labels_ith;
            std::string label;
            while(divid_sp >> seq) {
                labels_ith.push_back(seq);
            }
            debug_seq.push_back(labels_ith[0]);
            debug_seq_2.push_back(labels_ith[1]);
        }
    }

private:
    int num_frames_;
    int num_labels_;
    std::string label_path_;
    std::string pc_path_;
};

#endif //PATCHWORK_PCD_LOADER_HPP