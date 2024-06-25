# Archivo de configuración para latexmk
# https://mg.readthedocs.io/latexmk.html
# https://mg.readthedocs.io/latexmkrc.html
#
# Humberto Alejandro Ortega Alcocer
# 2023

@default_files = ('2024-A053.tex');

# Estoy usando Skim para poder actualizar en tiempo real.
$pdf_previewer = "open -a Skim";

# Para los glosarios y acrónimos
add_cus_dep( 'acn', 'acr', 0, 'makeglossaries' );
add_cus_dep( 'glo', 'gls', 0, 'makeglossaries' );
$clean_ext .= " acr acn alg glo gls glg";

sub makeglossaries {
   my ($base_name, $path) = fileparse( $_[0] );
   my @args = ( "-q", "-d", $path, $base_name );
   if ($silent) { unshift @args, "-q"; }
   return system "makeglossaries", "-d", $path, $base_name;
}
