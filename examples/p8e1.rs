extern crate cast;
extern crate posit;

use std::u8;

use cast::f64;
use posit::P8E1;

fn main() {
    for i in 0u16.. {
        if i > u8::MAX as u16 {
            break
        }

        let p = P8E1::new(i as u8);
        println!("{:3} - {}", i, f64(p));
    }
}
