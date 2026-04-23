$csv = Get-Content "data\manual\plantilla_completa_manual.csv" -Encoding UTF8 | ConvertFrom-Csv -Delimiter "`t"
$i = 1
$resultado = foreach ($fila in $csv) {
    [PSCustomObject]@{
        jugador_id = $i
        nombre = $fila.NOMBRE
        equipo = ""
        posicion = $fila.POSICION
        edad = $fila.EDAD -replace ' años',''
        nacionalidad = ""
        numero_camiseta = $fila.NUMERO
    }
    $i++
}
$resultado | Export-Csv "data\plantilla_maestra.csv" -NoTypeInformation -Encoding UTF8
Write-Host "Archivo generado correctamente"