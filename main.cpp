#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include "dextr.cpp"

int main() {
    find_dextr_bit_mask("../images/dog-cat.jpg",
            {{28, 205}, {42, 209}, {43, 187}, {69, 193}});

}