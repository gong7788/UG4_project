(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b1)
		(clear b1)
		(on-table b2)
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b4)
		(clear b4)
		(clear b6)
		(on-table b8)
		(clear b8)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b0 b9)
		(in-tower b0)
		(on b7 b0)
		(in-tower b7)
		(on b5 b7)
		(in-tower b5)
		(on b6 b5)
		(in-tower b6)
		(blue b0)
		(blue b3)
		(green b5)
		(red b7)
		(green b7)
		(maroon b7)
		(red b8)
		(maroon b8)
		(maroon b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?x) (or (not (green ?x)) (exists (?y) (and (maroon ?y) (on ?x ?y))))) (and (not (on b7 b9)) (not (on b3 b7)) (not (on b8 b7)) (not (on b6 b7)) (not (on b8 b5)) (not (on b8 b6)))))
)