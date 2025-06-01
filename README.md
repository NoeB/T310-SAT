## T310 SAT

Attempt to do a Known-plaintext attach of the T310 Streamcipher.

### Rustsat

Currently I am using a forc of RustSat which includes the CryptoMiniSat library
(based on the upstream branch) If CryptoMiniSat is not needed, then the upstream
version can be used. With CryptoMiniSat cmake and git must be installed on the
system.

## Credits

T310 Implementation:

- Schmeh, K. (2006). The East German Encryption Machine T-310 and the Algorithm
  It Used. Cryptologia, 30(3), 251–257.
  https://doi.org/10.1080/01611190600632457
- https://scz.bplaced.net/t310.html
- M. Altenhuber, Analyse und Implementierung der DDR-Chiffriermaschinen T-310/50
  und T-316
  (https://www.cryptool.org/media/publications/theses/BA_Altenhuber.pdf)
- C# T310 implementation:
  https://github.com/CrypToolProject/CrypTool-2/tree/386b778f1670c4def6b919f46ccecda3b055599c/CrypPlugins/T310
