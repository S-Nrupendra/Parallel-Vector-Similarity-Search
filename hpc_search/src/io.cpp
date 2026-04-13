/*
src/io.cpp — Loading the Binary File
This file tells How to read the .bin file into memory

opens the file in binary mode. ate means jumps to the end immediately so we can measure the file size
Each molecule takes 256 bytes. Divide total file size by 256 -> No. of molecules
*/
#include "fingerprint_db.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

FingerprintDB load_database(const std::string& path){
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if(!file.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    // Get file size
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Each molecule = FP_WORDS * 8 bytes
    size_t bytes_per_mol = FP_WORDS * sizeof(uint64_t); // 32 * 8 = 256 bytes
    size_t num_molecules = file_size / bytes_per_mol;

    std::cout<<"File size      : "<<file_size / (1024 * 1024)<<" MB\n";
    std::cout<<"Molecules found: "<<num_molecules<<"\n";

    FingerprintDB db;
    db.molecules.resize(num_molecules);

    // Read entire file in one shot
    file.read(
        reinterpret_cast<char*>(db.molecules.data()),
        num_molecules * bytes_per_mol
    );

    if(!file)
        throw std::runtime_error("Failed to read binary file completely");
    
    std::cout<<"Database loaded successfully.\n";
    return db;
}